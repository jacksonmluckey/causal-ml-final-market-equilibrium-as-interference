from pathlib import Path

def get_figures_dir():
    """Return the figures directory path."""
    return Path(__file__).parent.parent.parent / "figures"

def get_figures_path(filename, format="png"):
    """Return a path in the figures directory."""
    figures_dir = get_figures_dir()
    figures_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir / f"{filename}.{format}"