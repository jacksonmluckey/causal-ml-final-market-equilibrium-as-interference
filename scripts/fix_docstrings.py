"""
Convert docstrings to raw strings to fix LaTeX escape sequences.
Used for using LaTeX in MkDocStrings output.
"""

import re
from pathlib import Path


def fix_docstrings(file_path: Path) -> None:
    """
    Convert opening docstrings from \"\"\" to r\"\"\" while leaving closing \"\"\" unchanged.

    Args:
        file_path: Path to Python file to fix
    """
    content = file_path.read_text()

    # First, remove any existing 'r' prefix from ALL triple quotes
    content = re.sub(r'r"""', '"""', content)

    # Now add 'r' only to opening docstrings
    # Strategy: track whether we're inside a docstring or not

    lines = content.split('\n')
    result_lines = []
    in_docstring = False

    for line in lines:
        # Check if line contains """
        triple_quote_count = line.count('"""')

        if triple_quote_count == 0:
            # No triple quotes, just add the line
            result_lines.append(line)
        elif triple_quote_count == 2:
            # One-liner docstring: """something"""
            # This is both opening and closing
            # Add r prefix to opening
            new_line = re.sub(r'^(\s*)"""', r'\1r"""', line, count=1)
            result_lines.append(new_line)
        elif triple_quote_count == 1:
            # Either opening or closing
            if not in_docstring:
                # This is an opening docstring
                new_line = re.sub(r'^(\s*)"""', r'\1r"""', line)
                result_lines.append(new_line)
                in_docstring = True
            else:
                # This is a closing docstring
                result_lines.append(line)
                in_docstring = False
        else:
            # More than 2 triple quotes on one line - unusual, just add as-is
            result_lines.append(line)

    result_content = '\n'.join(result_lines)
    file_path.write_text(result_content)
    print(f"Fixed {file_path.name}")


def main():
    """Fix all Python files in src/demo/."""
    src_dir = Path(__file__).parent.parent / "src" / "demo"

    python_files = list(src_dir.glob("*.py"))

    for file_path in python_files:
        fix_docstrings(file_path)

    print(f"\nFixed {len(python_files)} files")


if __name__ == "__main__":
    main()
