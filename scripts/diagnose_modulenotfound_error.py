"""
Test to diagnose intermittent ModuleNotFoundError when importing demo.

This script verifies that uv run only reinstalls the editable package `demo`
when pyproject.toml is newer than the installation metadata, and that touching
pyproject.toml triggers a reinstall that fixes the import.
"""

import subprocess
from pathlib import Path


def run_python_import(code: str) -> tuple[bool, str]:
    """Run a Python command via uv run and return success status and output."""
    result = subprocess.run(
        ["uv", "run", "python", "-c", code],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0, result.stdout + result.stderr


def main():
    project_root = Path(__file__).parent.parent
    pyproject_path = project_root / "pyproject.toml"
    pth_file = list(
        project_root.glob(".venv/lib/python*/site-packages/__editable__.demo-*.pth")
    )

    print("=" * 60)
    print("Diagnostic Test for Intermittent ModuleNotFoundError")
    print("=" * 60)

    # Test 1: Try importing demo
    print("\n[Test 1] Attempting to import demo...")
    success, output = run_python_import("import demo; print('SUCCESS: demo imported')")

    if success:
        print("Success: Import succeeded on first try")
        print(f"Output: {output.strip()}")
        return

    print("Failure: Import failed")
    print(f"Error: {output.strip()}")

    # Test 2: Check sys.path
    print("\n[Test 2] Checking sys.path...")
    _, paths = run_python_import("import sys; print('\\n'.join(sys.path))")
    print("Python path entries:")
    for line in paths.strip().split("\n"):
        print(f"  {line}")

    # Test 3: Check .pth file
    print("\n[Test 3] Checking .pth file...")
    if pth_file:
        pth_file = pth_file[0]
        print(f"Found: {pth_file}")
        print(f"Contents: {pth_file.read_text().strip()}")
        print(f"Exists: {Path(pth_file.read_text().strip()).exists()}")
    else:
        print("Failure: No .pth file found!")

    # Test 4: Touch pyproject.toml and retry
    print("\n[Test 4] Touching pyproject.toml and retrying import...")
    pyproject_path.touch()
    print(f"Touched {pyproject_path}")

    success, output = run_python_import(
        "import demo; print('SUCCESS: demo imported after touch')"
    )

    if success:
        print("Import succeeded after touching pyproject.toml!")
        print(f"Output: {output.strip()}")
        print("\n" + "=" * 60)
        print("Touching pyproject.toml fixes the import issue.")
        print(
            "For whatever reason uv run only reinstalls when pyproject.toml is newer."
        )
        print("=" * 60)
    else:
        print("Failure: Import still failed after touching pyproject.toml")
        print(f"Error: {output.strip()}")
        print("\n" + "=" * 60)
        print("The ModuleNotFoundError is unrelated to pyproject.toml timestamps.")
        print("=" * 60)


if __name__ == "__main__":
    main()
