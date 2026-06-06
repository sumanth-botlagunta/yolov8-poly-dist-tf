"""Component registry for backbones, decoders, and heads.

Usage:
    from configs.registry import BACKBONES, DECODERS, HEADS

    @BACKBONES.register('cspdarknetv8s')
    class CSPDarkNetV8Small(...):
        ...

    backbone_cls = BACKBONES.get('cspdarknetv8s')
    backbone = backbone_cls(**kwargs)
"""

from typing import Callable, Dict, Type


class Registry:
    """Simple name → class registry with decorator support."""

    def __init__(self, name: str):
        self._name = name
        self._registry: Dict[str, Type] = {}

    def register(self, key: str) -> Callable:
        """Decorator that registers a class under the given key."""
        def decorator(cls: Type) -> Type:
            if key in self._registry:
                raise KeyError(
                    f"[{self._name}] Key '{key}' is already registered "
                    f"by {self._registry[key].__name__}."
                )
            self._registry[key] = cls
            return cls
        return decorator

    def get(self, key: str) -> Type:
        """Return the class registered under key, or raise KeyError."""
        if key not in self._registry:
            available = sorted(self._registry.keys())
            raise KeyError(
                f"[{self._name}] Key '{key}' not found. "
                f"Available: {available}"
            )
        return self._registry[key]

    def __contains__(self, key: str) -> bool:
        return key in self._registry

    def __repr__(self) -> str:
        keys = sorted(self._registry.keys())
        return f"Registry(name={self._name!r}, keys={keys})"


BACKBONES: Registry = Registry("BACKBONES")
DECODERS: Registry = Registry("DECODERS")
HEADS: Registry = Registry("HEADS")
