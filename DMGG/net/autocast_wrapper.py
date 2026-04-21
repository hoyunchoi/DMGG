import torch
from torch import nn

str_to_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
dtype_to_str = {torch.bfloat16: "bf16", torch.float16: "fp16", torch.float32: "fp32"}


class AutocastWrapper(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        dtype: str | torch.dtype = "bf16",
        device: torch.device | str = "cuda",
    ):
        super().__init__()
        self.module = module

        self._dtype = str_to_dtype[dtype] if isinstance(dtype, str) else dtype
        self._device_type = device.type if isinstance(device, torch.device) else device
        self._enable_amp = self._dtype != torch.float32

    def forward(self, *args, **kwargs):
        with torch.autocast(
            device_type=self._device_type, dtype=self._dtype, enabled=self._enable_amp
        ):
            return self.module(*args, **kwargs)

    def __getattr__(self, name: str):
        """
        Forward attribute access to the wrapped module if not found in wrapper
        """
        # Avoid infinite recursion by checking for 'module' first
        try:
            return super().__getattr__(name)
        except AttributeError:
            # If attribute not found in wrapper, try to get it from module
            module = super().__getattr__('module')
            return getattr(module, name)

    def state_dict(self, *args, **kwargs):
        return self.module.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, *args, **kwargs):
        return self.module.load_state_dict(state_dict, *args, **kwargs)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        return self.module._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def dtype_str(self) -> str:
        return dtype_to_str[self._dtype]
