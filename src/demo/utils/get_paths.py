from pathlib import Path


def get_project_root():
    """Returns the project root path"""
    return Path(__file__).parent.parent.parent.parent


def get_figures_dir():
    """Returns the figures directory path"""
    root = get_project_root()
    return root / "figures"


def get_data_dir():
    """Returns the data directory path"""
    root = get_project_root()
    return root / "data"


def get_figures_path(filename):
    """Return a path in the figures directory."""
    figures_dir = get_figures_dir()
    figures_dir.mkdir(exist_ok=True)
    return figures_dir / f"{filename}"


def get_data_path(filename):
    """Returns a path in the data directory"""
    data_dir = get_data_dir()
    data_dir.mkdir(exist_ok=True)
    return data_dir / f"{filename}"
