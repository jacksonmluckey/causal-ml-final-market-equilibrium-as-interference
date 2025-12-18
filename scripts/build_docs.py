"""
Build documentation for the demo package using MkDocs.
Can be run locally or in CI/CD.

Usage:
    uv run python scripts/build_docs.py
"""

import shutil
from pathlib import Path

from mkdocs.commands.build import build
from mkdocs.config import load_config


def generate_module_docs(src_dir: Path, api_dir: Path) -> list[str]:
    """
    Generate Markdown files for all Python modules in src/demo/.

    Args:
        src_dir: Path to src/demo/ directory
        api_dir: Path to docsrc/api/ directory

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

    # Create api directory if it doesn't exist
    api_dir.mkdir(parents=True, exist_ok=True)

    # Generate .md file for each module
    for module_name in modules:
        md_file = api_dir / f"{module_name}.md"

        # Create human-readable title from module name
        title = module_name.replace("_", " ").title()

        md_content = f"""# {title}

::: demo.{module_name}
"""

        md_file.write_text(md_content)
        print(f"  Generated {md_file.relative_to(api_dir.parent)}")

    return modules


def generate_mkdocs_config(project_root: Path, modules: list[str]) -> None:
    """
    Generate mkdocs.yml configuration file from template.

    Args:
        project_root: Path to project root directory
        modules: List of module names
    """
    template_file = project_root / "docsrc" / "mkdocs_config.yml"
    config_file = project_root / "docsrc" / "mkdocs.yml"

    # Read template
    template_content = template_file.read_text()

    # Build navigation structure
    nav_items = []
    for module_name in sorted(modules):
        title = module_name.replace("_", " ").title()
        nav_items.append(f"      - {title}: api/{module_name}.md")

    nav_section = "\n".join(nav_items)

    # Replace placeholder with generated navigation
    config_content = template_content.replace("{{NAV_ITEMS}}", nav_section)

    config_file.write_text(config_content)
    print(f"Generated {config_file.name}")


def main():
    """Build MkDocs documentation."""
    # Get project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    docsrc_dir = project_root / "docsrc"
    api_dir = docsrc_dir / "content" / "api"
    site_dir = project_root / "docs"
    src_dir = project_root / "src" / "demo"

    # Generate module documentation files
    modules = generate_module_docs(src_dir, api_dir)

    # Generate mkdocs.yml configuration
    # This allows for automatically discovering the modules in src/demo/
    generate_mkdocs_config(project_root, modules)

    # Clean previous build
    if site_dir.exists():
        shutil.rmtree(site_dir)

    # Build HTML documentation using MkDocs library
    config = load_config(config_file=str(docsrc_dir / "mkdocs.yml"))
    build(config)

    print(f"Open {site_dir / 'index.html'} to view the generated docs")


if __name__ == "__main__":
    main()
