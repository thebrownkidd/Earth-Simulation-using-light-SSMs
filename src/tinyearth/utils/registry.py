"""A small, typed component registry.

The central experimental claim of TinyEarth is that *only the temporal backbone
changes* between runs. That claim is only enforceable if swapping a backbone is
a configuration edit rather than a code edit. A registry gives us exactly that:
components declare themselves by name, and configs select them by name.

Why not Hydra's ``_target_`` instantiation, which does something similar? Two
reasons. First, ``_target_`` puts fully-qualified import paths in configs, so
renaming a module invalidates published experiment configs. Second, a registry
can be enumerated -- ``list(TEMPORAL_BACKBONES)`` is how the scaling sweep and
the docs discover what exists. The two mechanisms compose fine; this one is
preferred for the small set of swappable research components.

Example:
    >>> from tinyearth.utils.registry import Registry
    >>> backbones: Registry[object] = Registry("temporal_backbone")
    >>> @backbones.register()
    ... class MambaBackbone:
    ...     '''A backbone.'''
    >>> backbones.build("mamba_backbone")  # doctest: +ELLIPSIS
    <...MambaBackbone object at ...>
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator, Mapping
from typing import Generic, TypeVar

__all__ = ["DuplicateRegistrationError", "Registry", "UnknownComponentError"]

T = TypeVar("T")

_CAMEL_BOUNDARY = re.compile(r"(?<!^)(?=[A-Z])")


class UnknownComponentError(KeyError):
    """Raised when a name is not present in a registry."""


class DuplicateRegistrationError(ValueError):
    """Raised when a name is registered twice without ``override=True``."""


def _default_name(cls: type) -> str:
    """Derive a snake_case registry key from a class name.

    Args:
        cls: The class being registered.

    Returns:
        ``MambaBackbone`` becomes ``mamba_backbone``.
    """
    return _CAMEL_BOUNDARY.sub("_", cls.__name__).lower()


class Registry(Generic[T]):
    """A name-to-class mapping for one family of swappable components.

    Instances are module-level singletons, one per component family (encoders,
    temporal backbones, decoders, losses). The generic parameter is the common
    base class of the family, which keeps :meth:`build` return types precise.

    Attributes:
        name: Family name, used in error messages.
    """

    def __init__(self, name: str) -> None:
        """Create an empty registry.

        Args:
            name: Human-readable family name, e.g. ``"temporal_backbone"``.
        """
        self.name = name
        self._entries: dict[str, type[T]] = {}

    def register(
        self,
        key: str | None = None,
        *,
        override: bool = False,
    ) -> Callable[[type[T]], type[T]]:
        """Return a decorator that registers a class under ``key``.

        Args:
            key: Registry key. Defaults to the snake_case form of the class name.
            override: Permit replacing an existing entry. Off by default so that
                a duplicated name fails loudly rather than silently changing
                which architecture an experiment trains.

        Returns:
            A decorator returning its argument unchanged, so the decorated class
            remains directly importable and testable.
        """

        def decorator(cls: type[T]) -> type[T]:
            resolved = key if key is not None else _default_name(cls)
            if resolved in self._entries and not override:
                existing = self._entries[resolved]
                raise DuplicateRegistrationError(
                    f"{resolved!r} is already registered in the {self.name!r} registry "
                    f"as {existing.__module__}.{existing.__qualname__}. "
                    "Pass override=True if replacement is intended."
                )
            self._entries[resolved] = cls
            return cls

        return decorator

    def get(self, key: str) -> type[T]:
        """Look up a registered class.

        Args:
            key: Registry key.

        Returns:
            The registered class.

        Raises:
            UnknownComponentError: If ``key`` is absent. The message lists the
                available keys, since this failure almost always means a typo
                in a config file.
        """
        try:
            return self._entries[key]
        except KeyError:
            available = ", ".join(sorted(self._entries)) or "<none registered>"
            raise UnknownComponentError(
                f"{key!r} is not registered in the {self.name!r} registry. Available: {available}."
            ) from None

    def build(self, key: str, /, **kwargs: object) -> T:
        """Instantiate a registered class.

        Args:
            key: Registry key.
            **kwargs: Forwarded to the class constructor.

        Returns:
            The constructed instance.

        Raises:
            UnknownComponentError: If ``key`` is absent.
        """
        return self.get(key)(**kwargs)

    def keys(self) -> tuple[str, ...]:
        """Return registered keys in sorted order."""
        return tuple(sorted(self._entries))

    def as_mapping(self) -> Mapping[str, type[T]]:
        """Return a read-only view of the registry contents."""
        return dict(self._entries)

    def __contains__(self, key: object) -> bool:
        """Return whether ``key`` is registered."""
        return key in self._entries

    def __iter__(self) -> Iterator[str]:
        """Iterate over registered keys in sorted order."""
        return iter(self.keys())

    def __len__(self) -> int:
        """Return the number of registered components."""
        return len(self._entries)

    def __repr__(self) -> str:
        """Return a debugging representation listing the registered keys."""
        return f"Registry(name={self.name!r}, entries={list(self.keys())!r})"
