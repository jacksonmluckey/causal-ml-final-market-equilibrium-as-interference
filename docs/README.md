# Documentation

This directory contains the Sphinx documentation for the `demo` package implementing the marketplace equilibrium model from Wager & Xu (2021).

## Building Locally

1. Install documentation dependencies:
   ```bash
   uv sync --group docs
   ```

2. Build the documentation:
   ```bash
   uv run python scripts/build_docs.py
   ```

3. Open the generated documentation:
   ```bash
   open docs/_build/index.html
   ```

## Structure

- `conf.py` - Sphinx configuration with MathJax for LaTeX rendering
- `index.rst` - Main documentation page
- `modules/` - Individual module documentation pages
- `_build/` - Generated HTML documentation (not tracked in git)
- `_static/` - Static assets for documentation

## LaTeX Math

The documentation uses MathJax to render LaTeX math expressions from docstrings. Math can be written inline with `$...$` or in display mode with `$$...$$`.

## CI/CD

The GitHub Actions workflow (`.github/workflows/docs.yml`) automatically:
- Builds documentation on every push and pull request
- Deploys to GitHub Pages on pushes to main

To enable GitHub Pages deployment:
1. Go to repository Settings > Pages
2. Set Source to "GitHub Actions"
3. The documentation will be available at `https://<username>.github.io/<repo-name>/`

## Adding New Modules

To document a new module in `src/demo/`:

1. Create a new `.rst` file in `docs/modules/`:
   ```rst
   Module Name
   ===========

   .. automodule:: demo.module_name
      :members:
      :undoc-members:
      :show-inheritance:
   ```

2. Add it to the `toctree` in `docs/index.rst`:
   ```rst
   .. toctree::
      :maxdepth: 2

      modules/module_name
   ```

3. Rebuild the documentation
