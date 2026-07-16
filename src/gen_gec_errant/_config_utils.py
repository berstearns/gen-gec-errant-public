"""Shared config utilities: YAML loading, CLI overrides, serialization."""

import dataclasses
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar

import yaml

T = TypeVar("T")


def build_sub_config(config_cls: Type[T], raw: dict) -> T:
    """Instantiate a dataclass from a raw dict, ignoring unknown keys."""
    fields = {f.name for f in dataclasses.fields(config_cls)}
    filtered = {k: v for k, v in raw.items() if k in fields}
    return config_cls(**filtered)


def load_config_from_yaml(
    path: str | Path,
    config_cls: Type[T],
    section_map: Dict[str, Type] | None = None,
) -> T:
    """
    Load a YAML file and instantiate config_cls.

    section_map maps field names to sub-config dataclass types.
    e.g. {"generation": GenerationConfig, "gec": GECConfig}
    """
    path = Path(path)
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}

    return _build_config(raw, config_cls, section_map)


def _build_config(
    raw: dict,
    config_cls: Type[T],
    section_map: Dict[str, Type] | None = None,
) -> T:
    """Build a config from raw dict, handling nested sub-configs."""
    section_map = section_map or {}
    fields = {f.name for f in dataclasses.fields(config_cls)}
    kwargs: dict[str, Any] = {}

    for key, value in raw.items():
        if key not in fields:
            continue
        if key in section_map and isinstance(value, dict):
            kwargs[key] = build_sub_config(section_map[key], value)
        else:
            kwargs[key] = value

    return config_cls(**kwargs)


def apply_cli_overrides(
    config: T,
    overrides: list[str],
    section_map: Dict[str, Type] | None = None,
) -> T:
    """
    Apply dotted CLI overrides like ["generation.temperature=0.7", "batch_size=16"].

    Modifies config in-place and returns it.
    """
    section_map = section_map or {}

    for override in overrides:
        if "=" not in override:
            continue
        key, value = override.split("=", 1)
        # Strip leading -- if present
        key = key.lstrip("-")

        parts = key.split(".")
        if len(parts) == 2:
            section, field = parts
            sub = getattr(config, section, None)
            if sub is not None and dataclasses.is_dataclass(sub):
                _set_field(sub, field, value)
        elif len(parts) == 1:
            _set_field(config, parts[0], value)

    return config


def _set_field(obj: Any, field_name: str, value_str: str) -> None:
    """Set a dataclass field, casting to the field's declared type."""
    if not hasattr(obj, field_name):
        return

    # Find the field type
    for f in dataclasses.fields(obj):
        if f.name == field_name:
            field_type = f.type
            break
    else:
        return

    # Cast value
    parsed = _cast_value(value_str, field_type)
    setattr(obj, field_name, parsed)


def _cast_value(value_str: str, type_hint: Any) -> Any:
    """Cast a string to the appropriate Python type."""
    # Handle string representations
    type_str = str(type_hint)

    if type_str == "bool" or type_hint is bool:
        return value_str.lower() in ("true", "1", "yes")
    if type_str == "int" or type_hint is int:
        return int(value_str)
    if type_str == "float" or type_hint is float:
        return float(value_str)
    if type_str == "str" or type_hint is str:
        return value_str
    # Optional types — try int, then float, then str
    if "Optional" in type_str or "None" in type_str:
        if value_str.lower() == "none":
            return None
        try:
            return int(value_str)
        except ValueError:
            pass
        try:
            return float(value_str)
        except ValueError:
            pass
        return value_str

    return value_str


def config_to_yaml(config: Any) -> str:
    """Serialize a dataclass config to YAML string."""
    return yaml.dump(
        _dataclass_to_dict(config),
        default_flow_style=False,
        sort_keys=False,
    )


def _dataclass_to_dict(obj: Any) -> Any:
    """Recursively convert a dataclass to dict."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {
            f.name: _dataclass_to_dict(getattr(obj, f.name))
            for f in dataclasses.fields(obj)
        }
    if isinstance(obj, list):
        return [_dataclass_to_dict(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _dataclass_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, Path):
        return str(obj)
    return obj
