"""Tests for the component registry."""

from __future__ import annotations

import pytest

from tinyearth.utils.registry import (
    DuplicateRegistrationError,
    Registry,
    UnknownComponentError,
)


class Base:
    """Base class for registry test doubles."""

    def __init__(self, value: int = 0) -> None:
        self.value = value


@pytest.fixture
def registry() -> Registry[Base]:
    return Registry[Base]("test_family")


def test_register_derives_snake_case_name(registry):
    @registry.register()
    class MambaBackbone(Base):
        """Test double."""

    assert "mamba_backbone" in registry
    assert registry.get("mamba_backbone") is MambaBackbone


def test_register_accepts_explicit_key(registry):
    @registry.register("s4")
    class S4Backbone(Base):
        """Test double."""

    assert registry.keys() == ("s4",)


def test_decorator_returns_the_class_unchanged(registry):
    @registry.register()
    class Thing(Base):
        """Test double."""

    assert Thing.__name__ == "Thing"
    assert issubclass(Thing, Base)


def test_duplicate_registration_is_rejected(registry):
    @registry.register("dup")
    class First(Base):
        """Test double."""

    with pytest.raises(DuplicateRegistrationError, match="already registered"):

        @registry.register("dup")
        class Second(Base):
            """Test double."""


def test_override_allows_replacement(registry):
    @registry.register("dup")
    class First(Base):
        """Test double."""

    @registry.register("dup", override=True)
    class Second(Base):
        """Test double."""

    assert registry.get("dup") is Second


def test_unknown_key_lists_available_options(registry):
    @registry.register("alpha")
    class Alpha(Base):
        """Test double."""

    with pytest.raises(UnknownComponentError, match="alpha"):
        registry.get("typo")


def test_unknown_key_on_empty_registry(registry):
    with pytest.raises(UnknownComponentError, match="<none registered>"):
        registry.get("anything")


def test_build_forwards_kwargs(registry):
    @registry.register("thing")
    class Thing(Base):
        """Test double."""

    instance = registry.build("thing", value=17)
    assert isinstance(instance, Thing)
    assert instance.value == 17


def test_keys_are_sorted(registry):
    for name in ("zulu", "alpha", "mike"):
        registry.register(name)(type(name.title(), (Base,), {"__doc__": "d"}))

    assert registry.keys() == ("alpha", "mike", "zulu")


def test_len_and_iteration(registry):
    registry.register("a")(type("A", (Base,), {"__doc__": "d"}))
    registry.register("b")(type("B", (Base,), {"__doc__": "d"}))

    assert len(registry) == 2
    assert list(registry) == ["a", "b"]


def test_as_mapping_is_a_copy(registry):
    registry.register("a")(type("A", (Base,), {"__doc__": "d"}))
    mapping = registry.as_mapping()
    dict(mapping).clear()

    assert "a" in registry
