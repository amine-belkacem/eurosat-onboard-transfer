"""From a training checkpoint to ONNX models (FP32 and INT8) that take raw tiles.

The exported graph includes the preprocessing: its input is the uint8 tile as
it comes off the sensor product (N, 3, 64, 64), and resizing, scaling and
normalization run inside ONNX Runtime. Two reasons:

* the Raspberry Pi side needs no PyTorch and cannot drift from the training
  preprocessing (bicubic resizing differs between libraries);
* the latency I measure is the latency of the whole tile -> class step, not of
  the network alone.

INT8 uses ONNX Runtime static quantization in QDQ format: weights per channel
(int8), activations per tensor (uint8), scales calibrated on validation tiles.
Only Conv, MatMul and Gemm are quantized. Resize, LayerNorm, Softmax and GELU
stay in float, which keeps the preprocessing exact and avoids the operators
that usually lose the most accuracy in transformers.
"""

from __future__ import annotations

import hashlib
import warnings
from pathlib import Path

import numpy as np
import torch
from torch import nn

from eot import data, models

ONNX_OPSET = 17
INPUT_NAME = "tiles"
OUTPUT_NAME = "logits"
QUANTIZED_OPS = ["Conv", "MatMul", "Gemm"]


class TileClassifier(nn.Module):
    """uint8 tiles (N, 3, 64, 64) -> class logits, preprocessing included."""

    def __init__(self, model: nn.Module, input_size: int):
        super().__init__()
        self.model = model
        self.input_size = input_size

    def forward(self, tiles: torch.Tensor) -> torch.Tensor:
        return self.model(data.preprocess(tiles, self.input_size))


def load_checkpoint(path: Path) -> tuple[TileClassifier, dict]:
    """Rebuild the trained model from a checkpoint written by experiment.save_checkpoint."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = models.create_model(ckpt["model"], pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    meta = {k: v for k, v in ckpt.items() if k != "state_dict"}
    return TileClassifier(model, ckpt["input_size"]).eval(), meta


def export_onnx(module: TileClassifier, path: Path) -> None:
    """Export with a dynamic batch dimension (batch 1 for latency, larger for accuracy)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    example = torch.zeros(1, 3, data.NATIVE_SIZE, data.NATIVE_SIZE, dtype=torch.uint8)
    # Legacy TorchScript exporter (dynamo=False): PyTorch marks it deprecated, but
    # the newer exporter needs onnxscript and this one is checked against
    # PyTorch by check_equivalence on every export. The tracer warns about the
    # Python `if` in preprocess(); the branch only depends on the fixed input
    # size, so tracing it once is correct.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", torch.jit.TracerWarning)
        torch.onnx.export(module.eval(), (example,), str(path), input_names=[INPUT_NAME],
                          output_names=[OUTPUT_NAME], opset_version=ONNX_OPSET,
                          dynamic_axes={INPUT_NAME: {0: "n"}, OUTPUT_NAME: {0: "n"}}, dynamo=False)


def onnx_parameter_count(path: Path) -> int:
    import onnx

    graph = onnx.load(str(path)).graph
    return int(sum(np.prod(init.dims) for init in graph.initializer))


def check_equivalence(module: TileClassifier, onnx_path: Path, tiles: np.ndarray) -> dict:
    """Compare PyTorch and ONNX Runtime FP32 logits on the same tiles."""
    import onnxruntime as ort

    with torch.no_grad():
        ref = module(torch.from_numpy(tiles)).numpy()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    out = sess.run(None, {INPUT_NAME: tiles})[0]
    return {
        "max_abs_diff": float(np.abs(out - ref).max()),
        "argmax_agreement": float((out.argmax(1) == ref.argmax(1)).mean()),
        "n_tiles": int(len(tiles)),
    }


class _TileReader:
    """Feeds calibration tiles to the quantizer one batch at a time."""

    def __init__(self, tiles: np.ndarray, batch_size: int):
        self._batches = iter([tiles[i:i + batch_size] for i in range(0, len(tiles), batch_size)])

    def get_next(self):
        batch = next(self._batches, None)
        return None if batch is None else {INPUT_NAME: batch}


def quantize_int8(fp32_path: Path, int8_path: Path, calibration_tiles: np.ndarray,
                  batch_size: int = 32) -> None:
    """Static INT8 quantization (QDQ, per-channel weights) calibrated on `calibration_tiles`."""
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    class Reader(_TileReader, CalibrationDataReader):
        pass

    pre_path = int8_path.with_name(int8_path.stem + "_pre.onnx")
    quant_pre_process(str(fp32_path), str(pre_path))
    try:
        quantize_static(str(pre_path), str(int8_path), Reader(calibration_tiles, batch_size),
                        quant_format=QuantFormat.QDQ, per_channel=True,
                        weight_type=QuantType.QInt8, activation_type=QuantType.QUInt8,
                        op_types_to_quantize=QUANTIZED_OPS)
    finally:
        pre_path.unlink(missing_ok=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
