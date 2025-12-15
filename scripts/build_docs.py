#!/usr/bin/env python
"""
Build documentation for the demo package.
Can be run locally or in CI/CD.

Usage:
    uv run python scripts/build_docs.py
"""

import shutil
import subprocess
from pathlib import Path


def generate_module_docs(src_dir: Path, modules_dir: Path) -> list[str]:
    """
    Generate .rst files for all Python modules in src/demo/.

    Args:
        src_dir: Path to src/demo/ directory
        modules_dir: Path to docs/modules/ directory

    Returns:
        List of module names (without .py extension)
    """
    # Find all Python files in src/demo/
    python_files = sorted(src_dir.glob("*.py"))

    # Filter out __init__.py and __pycache__
    modules = [
        f.stem for f in python_files
        if f.stem != "__init__" and not f.stem.startswith("_")
    ]

    print(f"Found {len(modules)} modules: {', '.join(modules)}")

    # Create modules directory if it doesn't exist
    modules_dir.mkdir(parents=True, exist_ok=True)

    # Generate .rst file for each module
    for module_name in modules:
        rst_file = modules_dir / f"{module_name}.rst"

        # Create human-readable title from module name
        title = module_name.replace("_", " ").title()
        title_underline = "=" * len(title)

        rst_content = f"""{title}
{title_underline}

.. automodule:: demo.{module_name}
   :members:
   :undoc-members:
   :show-inheritance:
"""

        rst_file.write_text(rst_content)
        print(f"  Generated {rst_file.relative_to(modules_dir.parent)}")

    return modules


def update_index_toctree(docs_dir: Path, modules: list[str]) -> None:
    """
    Update the toctree in index.rst with all modules.

    Args:
        docs_dir: Path to docs/ directory
        modules: List of module names
    """
    index_file = docs_dir / "index.rst"

    # Build toctree entries
    toctree_entries = "\n   ".join(f"modules/{mod}" for mod in sorted(modules))

    toctree_section = f"""API Reference
-------------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   {toctree_entries}"""

    # Read current index
    content = index_file.read_text()

    # Replace the API Reference section
    import re
    pattern = r"API Reference\n-+\n\n\.\.  toctree::.*?(?=\n\n[A-Z]|\Z)"

    if re.search(pattern, content, re.DOTALL):
        new_content = re.sub(pattern, toctree_section, content, flags=re.DOTALL)
        index_file.write_text(new_content)
        print(f"Updated {index_file.relative_to(docs_dir.parent)}")
    else:
        print(f"Warning: Could not find API Reference section in {index_file}")


def main():
    """Build Sphinx documentation."""
    # Get project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    docs_dir = project_root / "docs"
    modules_dir = docs_dir / "modules"
    build_dir = docs_dir / "_build"
    src_dir = project_root / "src" / "demo"

    print("Generating documentation...")

    # Generate module documentation files
    print("\nGenerating module .rst files...")
    modules = generate_module_docs(src_dir, modules_dir)

    # Update index.rst with all modules
    print("\nUpdating index.rst...")
    update_index_toctree(docs_dir, modules)

    # Clean previous build
    if build_dir.exists():
        print(f"\nCleaning {build_dir}")
        shutil.rmtree(build_dir)

    # Build HTML documentation
    print("\nRunning Sphinx build...")
    subprocess.run(
        ["python", "-m", "sphinx", "-b", "html", str(docs_dir), str(build_dir)],
        check=True,
    )

    print("\nDocumentation built successfully!")
    print(f"Open {build_dir / 'index.html'} to view")


if __name__ == "__main__":
    main()
