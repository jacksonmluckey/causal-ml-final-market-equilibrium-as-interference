#!/usr/bin/env python
"""Convert inline $...$ math to Sphinx :math:`...` roles in docstrings."""

import re
from pathlib import Path


def convert_math_in_docstrings(file_path: Path) -> None:
    """
    Convert $...$ to :math:`...` in docstrings for proper Sphinx rendering.

    Args:
        file_path: Path to Python file to fix
    """
    content = file_path.read_text()

    lines = content.split('\n')
    result_lines = []
    in_docstring = False

    for line in lines:
        # Check if we're entering/exiting a docstring
        triple_quote_count = line.count('"""')

        if triple_quote_count == 1:
            if not in_docstring:
                in_docstring = True
            else:
                in_docstring = False
            result_lines.append(line)
        elif triple_quote_count == 2:
            # One-liner docstring - process it
            new_line = convert_inline_math(line)
            result_lines.append(new_line)
        elif in_docstring:
            # We're inside a docstring - convert math
            new_line = convert_inline_math(line)
            result_lines.append(new_line)
        else:
            # Not in docstring - don't touch it
            result_lines.append(line)

    result_content = '\n'.join(result_lines)
    file_path.write_text(result_content)
    print(f"Converted math in {file_path.name}")


def convert_inline_math(line: str) -> str:
    """
    Convert $...$ to :math:`...` in a single line.

    Handles both inline math and keeps display math as is for now.
    """
    # Don't convert $$ (display math) - only single $
    # Pattern: $ followed by non-$ content, then closing $
    # Use negative lookahead/lookbehind to avoid $$

    # Match $...$ but not $$...$$
    # Strategy: match $ not followed by $, then content, then $ not preceded by $
    pattern = r'(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)'

    def replace_math(match):
        math_content = match.group(1)
        # Escape backticks in math content
        math_content = math_content.replace('`', r'\`')
        return f':math:`{math_content}`'

    new_line = re.sub(pattern, replace_math, line)

    return new_line


def main():
    """Convert math in all Python files in src/demo/."""
    src_dir = Path(__file__).parent.parent / "src" / "demo"

    python_files = sorted(src_dir.glob("*.py"))

    for file_path in python_files:
        if file_path.stem != "__init__":  # Skip __init__.py
            convert_math_in_docstrings(file_path)

    print(f"\nConverted {len(python_files) - 1} files")


if __name__ == "__main__":
    main()
