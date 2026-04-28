"""Nahual server for CellDino.

CellDino is a transformer-based cell segmentation/tracking model based on
MaskDINO. The full ``CtDINO`` meta-architecture pulls in a CUDA-compiled
MSDeformAttn op (built via ``maskdino/modeling/pixel_decoder/ops/make.sh``)
and the heavier mmcv/pycocotools-based criterion stack, neither of which is
straightforward to bring up under nix without a custom CUDA build step.

For the smoke test we therefore expose just the **backbone** of CellDino —
either the default ResNet-50 backbone (matching ``Base-Cell-InstanceSegmentation``)
or, optionally, the ``D2SwinTransformer`` defined in
``maskdino/modeling/backbone/swin.py``. The backbone alone is enough to verify
the server contract: it accepts a 5D ``NCZYX`` numpy array, drops Z, runs the
backbone, and returns a feature map (highest-resolution res5 by default).

This mirrors the pattern used in ``scdino/server.py`` and ``dinov3/server.py``.

Run with:
    nix run --impure . -- ipc:///tmp/celldino.ipc
or:
    python server.py ipc:///tmp/celldino.ipc
"""

import os
import sys
from functools import partial
from typing import Callable

import numpy
import pynng
import torch
import trio
from nahual.preprocess import pad_channel_dim, validate_input_shape
from nahual.server import responder

# Make the local ``maskdino`` package importable so its backbone registers
# itself with detectron2's ``BACKBONE_REGISTRY``.
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from detectron2.config import get_cfg  # noqa: E402
from detectron2.modeling import build_backbone  # noqa: E402

# NOTE on backbones: CellDino's repo registers a custom ``D2SwinTransformer``
# in ``maskdino/modeling/backbone/swin.py``, but importing that module pulls in
# ``timm`` (its pytest suite is extremely slow to build under nix). We default
# to detectron2's built-in ``build_resnet_backbone`` (matching the
# ``Base-Cell-InstanceSegmentation.yaml`` config from CellDino) which has zero
# extra deps. Set ``backbone="swin"`` to opt into the local Swin module — that
# requires ``timm`` to be available at runtime.
address = sys.argv[1]


def _make_default_cfg(backbone: str) -> "object":
    """Build a minimal detectron2 config for the requested backbone."""
    cfg = get_cfg()
    if backbone == "swin":
        # Swin-T defaults pulled from add_maskdino_config in maskdino/config.py.
        # Importing here (lazy) so we don't pay the timm cost unless requested.
        from detectron2.config import CfgNode as CN
        from maskdino.modeling.backbone import swin as _swin  # noqa: F401

        cfg.MODEL.BACKBONE.NAME = "D2SwinTransformer"
        cfg.MODEL.SWIN = CN()
        cfg.MODEL.SWIN.PRETRAIN_IMG_SIZE = 224
        cfg.MODEL.SWIN.PATCH_SIZE = 4
        cfg.MODEL.SWIN.EMBED_DIM = 96
        cfg.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
        cfg.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
        cfg.MODEL.SWIN.WINDOW_SIZE = 7
        cfg.MODEL.SWIN.MLP_RATIO = 4.0
        cfg.MODEL.SWIN.QKV_BIAS = True
        cfg.MODEL.SWIN.QK_SCALE = None
        cfg.MODEL.SWIN.DROP_RATE = 0.0
        cfg.MODEL.SWIN.ATTN_DROP_RATE = 0.0
        cfg.MODEL.SWIN.DROP_PATH_RATE = 0.3
        cfg.MODEL.SWIN.APE = False
        cfg.MODEL.SWIN.PATCH_NORM = True
        cfg.MODEL.SWIN.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
        cfg.MODEL.SWIN.USE_CHECKPOINT = False
    elif backbone == "resnet":
        # Match Base-Cell-InstanceSegmentation.yaml.
        cfg.MODEL.BACKBONE.NAME = "build_resnet_backbone"
        cfg.MODEL.RESNETS.DEPTH = 50
        cfg.MODEL.RESNETS.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
        cfg.MODEL.RESNETS.STEM_OUT_CHANNELS = 64
        cfg.MODEL.RESNETS.STRIDE_IN_1X1 = False
    else:
        raise ValueError(
            f"Unknown backbone {backbone!r}; expected 'swin' or 'resnet'"
        )

    cfg.MODEL.PIXEL_MEAN = [127.5, 127.5, 127.5]
    cfg.MODEL.PIXEL_STD = [57.0, 57.0, 57.0]
    return cfg


def setup(
    backbone: str = "resnet",
    weights: str | None = None,
    device: int | None = None,
    expected_tile_size: int = 32,
    expected_channels: int = 3,
    feature_key: str = "res5",
) -> tuple[Callable, dict]:
    """Build the CellDino backbone (random init unless ``weights`` is provided).

    Parameters
    ----------
    backbone : str
        Backbone name: ``"resnet"`` (default, R-50) or ``"swin"`` (Swin-T).
    weights : str | None
        Path to a state-dict-style checkpoint with backbone keys. If None the
        backbone uses random initialisation — fine for the smoke test.
    device : int | None
        CUDA device index. None → cuda:0 if available, else cpu.
    expected_tile_size : int
        Input H/W must be divisible by this. 32 covers both ResNet-50 and
        Swin-T (which need 32-divisible spatial dims).
    expected_channels : int
        Number of input channels expected by the backbone (3 for both).
    feature_key : str
        Which output feature map to return — one of ``res2``/``res3``/``res4``/``res5``.
    """
    if device is None:
        device = 0
    if torch.cuda.is_available():
        torch_device = torch.device(int(device))
    else:
        torch_device = torch.device("cpu")

    cfg = _make_default_cfg(backbone)
    model = build_backbone(cfg)

    load_info = {"weights": "random"}
    if weights is not None and os.path.exists(weights):
        state_dict = torch.load(weights, map_location="cpu")
        if isinstance(state_dict, dict):
            for k in ("model", "state_dict", "teacher", "student"):
                if k in state_dict:
                    state_dict = state_dict[k]
                    break
        # Strip common backbone prefixes.
        cleaned = {}
        for k, v in state_dict.items():
            kk = k.replace("module.", "")
            if kk.startswith("backbone."):
                kk = kk[len("backbone.") :]
            cleaned[kk] = v
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        load_info = {
            "missing": len(missing),
            "unexpected": len(unexpected),
            "weights": weights,
        }

    model.to(torch_device).eval()

    info = {
        "device": str(torch_device),
        "backbone": backbone,
        "feature_key": feature_key,
        "load": load_info,
    }
    processor = partial(
        process,
        model=model,
        device=torch_device,
        expected_tile_size=expected_tile_size,
        expected_channels=expected_channels,
        feature_key=feature_key,
    )
    return processor, info


def process(
    pixels: numpy.ndarray,
    model,
    device: torch.device,
    expected_tile_size: int,
    expected_channels: int,
    feature_key: str,
) -> torch.Tensor:
    """Forward an NCZYX numpy array through the CellDino backbone.

    Returns the requested feature map (``res5`` by default), shape
    ``(N, C_out, H/stride, W/stride)``. The Nahual responder converts torch
    tensors → numpy automatically.
    """
    if pixels.ndim != 5:
        raise ValueError(
            f"Expected NCZYX (5D) array, got shape {pixels.shape}"
        )
    _, _, _, *input_yx = pixels.shape
    validate_input_shape(input_yx, expected_tile_size)

    # ``pad_channel_dim`` drops the Z axis (returns NCYX) and pads channels
    # up to ``expected_channels`` if needed.
    pixels = pad_channel_dim(pixels, expected_channels)

    torch_tensor = torch.from_numpy(pixels.copy()).float().to(device)

    with torch.no_grad():
        features = model(torch_tensor)

    # detectron2 backbones return a dict[str, Tensor] keyed by stage name.
    if isinstance(features, dict):
        if feature_key in features:
            return features[feature_key]
        # Fall back to whatever the deepest stage is.
        return features[sorted(features)[-1]]
    return features


async def main():
    with pynng.Rep0(listen=address, recv_timeout=300) as sock:
        print(f"CellDino server listening on {address}", flush=True)
        async with trio.open_nursery() as nursery:
            responder_curried = partial(responder, setup=setup)
            nursery.start_soon(responder_curried, sock)


if __name__ == "__main__":
    try:
        trio.run(main)
    except KeyboardInterrupt:
        pass
