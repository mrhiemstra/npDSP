from __future__ import annotations

import inspect
from collections.abc import Callable
from pathlib import Path
from typing import cast

import npdsp

SOURCE = Path(__file__).parent
GENERATED = SOURCE / "generated"
API = SOURCE / "api.rst"


def get_submodule(obj: object) -> str:
    module = getattr(obj, "__module__", "npdsp")

    if module.startswith("npdsp."):
        module = module.removeprefix("npdsp.")

    return module


def get_category(obj: object) -> str:
    module = get_submodule(obj)

    if module.startswith("core"):
        return "Core"

    if module.startswith("blocks"):
        return "Blocks"

    if module.startswith("math"):
        return "Math"

    if module.startswith("conversion"):
        return "Conversion"

    return "Other"


def format_signature(obj: object) -> str:
    try:
        return str(inspect.signature(cast(Callable[..., object], obj)))
    except (TypeError, ValueError):
        return ""


def generate_object_page(name: str, obj: object) -> None:
    path = GENERATED / f"{name}.rst"

    title = f"{name}"
    underline = "=" * len(title)

    lines = [
        title,
        underline,
        "",
        ".. currentmodule:: npdsp",
        "",
        f".. autoclass:: npdsp.{name}",
        "   :members:",
        "   :undoc-members:",
        "",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")


def generate() -> None:
    GENERATED.mkdir(parents=True, exist_ok=True)

    groups: dict[str, list[str]] = {}

    for name in npdsp.__all__:
        obj = getattr(npdsp, name)

        if not (inspect.isclass(obj) or inspect.isfunction(obj)):
            continue

        category = get_category(obj)
        groups.setdefault(category, []).append(name)

        generate_object_page(name, obj)

    lines = [
        "API Reference",
        "=============",
        "",
        "The public npDSP API.",
        "",
    ]

    category_order = [
        "Core",
        "Blocks",
        "Math",
        "Conversion",
        "Other",
    ]

    for category in category_order:
        names = groups.get(category)

        if not names:
            continue

        lines.extend(
            [
                category,
                "-" * len(category),
                "",
                ".. toctree::",
                "   :maxdepth: 1",
                "",
            ]
        )

        for name in sorted(names):
            lines.append(f"   generated/{name}")

        lines.append("")

    API.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    generate()
