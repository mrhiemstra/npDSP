from __future__ import annotations

import inspect
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from shutil import rmtree
from typing import cast

import npdsp

SOURCE = Path(__file__).parent
GENERATED = SOURCE / "generated"
GENERATED_API = GENERATED / "api"
EXAMPLES_SOURCE = Path(__file__).parent.parent.parent / "examples"
GENERATED_EXAMPLES = GENERATED / "examples"

API = SOURCE / "api.rst"
EXAMPLES = SOURCE / "examples.rst"


def get_submodule(obj: object) -> str:
    if inspect.ismodule(obj):
        module = obj.__name__
    else:
        module = obj.__module__

    if module.startswith("npdsp."):
        module = module.removeprefix("npdsp.")

    return module


def get_category(obj: object) -> str:
    module = get_submodule(obj)

    if module.startswith("core"):
        return "Core"

    if module.startswith("blocks"):
        return "Blocks"

    if module.startswith("npdsp"):
        return "npdsp"

    return "Other"


def format_signature(obj: object) -> str:
    try:
        return str(inspect.signature(cast(Callable[..., object], obj)))
    except (TypeError, ValueError):
        return ""


def generate_object_page(name: str, obj: object) -> None:
    path = GENERATED_API / f"{name}.rst"

    if inspect.ismodule(obj):
        class_name = obj.__name__
    else:
        class_name = f"npdsp.{name}"

    title = f"{name}"
    underline = "=" * len(title)

    lines = [
        title,
        underline,
        "",
        ".. currentmodule:: npdsp",
        f".. {'automodule' if inspect.ismodule(obj) else 'autoclass'}:: {class_name}",
        "   :members:",
        "   :undoc-members:",
        "",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")


def generate_api() -> None:
    if GENERATED_API.exists() and GENERATED_API.is_dir():
        rmtree(GENERATED_API)

    GENERATED_API.mkdir(parents=True, exist_ok=False)

    groups: dict[str, list[str]] = {}

    for name in npdsp.__all__:
        obj = getattr(npdsp, name)

        if not (
            inspect.isclass(obj) or inspect.isfunction(obj) or inspect.ismodule(obj)
        ):
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
            lines.append(f"   generated/api/{name}")

        lines.append("")

    API.write_text("\n".join(lines), encoding="utf-8")


def generate_examples() -> None:
    if GENERATED_EXAMPLES.exists() and GENERATED_EXAMPLES.is_dir():
        rmtree(GENERATED_EXAMPLES)

    GENERATED_EXAMPLES.mkdir(parents=True, exist_ok=False)

    if EXAMPLES.exists() and EXAMPLES.is_file():
        EXAMPLES.unlink()

    # Create examples.rst
    lines = [
        "Examples",
        "========",
        "",
        "The following examples demonstrate how to use npDSP.",
        "",
        ".. toctree::",
        "   :maxdepth: 1",
        "",
        "",
    ]

    EXAMPLES.write_text("\n".join(lines), encoding="utf-8")

    for example_file in EXAMPLES_SOURCE.glob("*.py"):
        example_name = example_file.stem
        generated_example_file = GENERATED_EXAMPLES / f"{example_name}.rst"


        lines = [
            example_name,
            "=" * len(example_name),
            "Code",
            "----",
            "",
            ".. code-block:: python",
            "   :linenos:",
            "", 
        ]


        with example_file.open("r", encoding="utf-8") as f:
            for line in f:
                lines.append(f"   {line.rstrip()}")

        lines += [
            "",
            "Output",
            "------",
            "",
            ".. code-block:: text",
            "",
        ]

        result = subprocess.run(
            [sys.executable, str(example_file)],
            capture_output=True,
            text=True,
            cwd=EXAMPLES_SOURCE,
        ).stdout.splitlines()

        for line in result:
            lines.append(f"   {line.rstrip()}")

        generated_example_file.write_text("\n".join(lines), encoding="utf-8")

        with EXAMPLES.open("a", encoding="utf-8") as f:
            f.write(f"   generated/examples/{example_name}\n")

def generate_getting_started() -> None:
    getting_started_file = SOURCE / "getting_started.rst"

    if getting_started_file.exists() and getting_started_file.is_file():
        getting_started_file.unlink()

    with open(SOURCE.parent.parent / "README.rst", "r", encoding="utf-8") as f:
        lines = f.readlines()

    getting_started_file.write_text("".join(lines), encoding="utf-8")

if __name__ == "__main__":
    generate_api()
    generate_examples()
    generate_getting_started()
