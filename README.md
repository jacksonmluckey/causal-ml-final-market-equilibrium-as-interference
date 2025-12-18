# Markets as Interference: Experimenting in Equilibrium

Implements [_Experimenting in Equilibrium_ by Stefen Wager and Kuang Xu](https://doi.org/10.48550/arXiv.1903.02124)

## Docs

I generate documentation using MkDocs from the docstrings in demo. The docs are stored in `docs/`.

This process uses MathJax to render LaTeX math expressions from docstrings (written inline with `$...$` and in display mode with `$$...$$`).


`scripts/build_docs.py` automatically adds all modules in `src/demo/` to the documentation.

`docsrc/` contains the configuration for the docs. `mkdocs.yml` is automatically generated--use `mkdocs_config.yml`.

The documentation dependencies can be installed with `uv sync --group docs`.

The generation of the docs is part of a GitHub Actions workflow (`.github/workflows/docs.yml`).

You can serve the documentation with live reload of the already-discovered modules with:

```bash
uv run mkdocs serve
```

Then open http://127.0.0.1:8000 in your browser.