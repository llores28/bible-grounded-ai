"""CUDA capability checks for local QLoRA training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class CudaDevice:
    index: int
    name: str
    vram_gib: float
    compute_capability: str


@dataclass(frozen=True, slots=True)
class CudaReport:
    torch_installed: bool
    cuda_available: bool
    torch_version: str | None
    cuda_runtime: str | None
    bf16_supported: bool
    devices: tuple[CudaDevice, ...]
    meets_minimum_vram: bool
    minimum_vram_gib: float
    error: str | None = None

    @property
    def ready(self) -> bool:
        return (
            self.torch_installed
            and self.cuda_available
            and self.bf16_supported
            and self.meets_minimum_vram
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def inspect_cuda(*, minimum_vram_gib: float = 24.0) -> CudaReport:
    try:
        import torch
    except ImportError:
        return CudaReport(
            False,
            False,
            None,
            None,
            False,
            (),
            False,
            minimum_vram_gib,
            "PyTorch is not installed; install the training extra with a CUDA-compatible PyTorch build.",
        )

    if not torch.cuda.is_available():
        return CudaReport(
            True,
            False,
            torch.__version__,
            getattr(torch.version, "cuda", None),
            False,
            (),
            False,
            minimum_vram_gib,
            "torch.cuda.is_available() returned false.",
        )

    devices = tuple(
        CudaDevice(
            index=index,
            name=torch.cuda.get_device_name(index),
            vram_gib=round(torch.cuda.get_device_properties(index).total_memory / (1024**3), 2),
            compute_capability=".".join(
                str(value) for value in torch.cuda.get_device_capability(index)
            ),
        )
        for index in range(torch.cuda.device_count())
    )
    bf16_supported = bool(torch.cuda.is_bf16_supported())
    return CudaReport(
        True,
        True,
        torch.__version__,
        getattr(torch.version, "cuda", None),
        bf16_supported,
        devices,
        any(device.vram_gib >= minimum_vram_gib for device in devices),
        minimum_vram_gib,
    )
