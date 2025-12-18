# Markets as Interference: Experimenting in Equilibrium

Implements [_Experimenting in Equilibrium_ by Stefen Wager and Kuang Xu](https://doi.org/10.48550/arXiv.1903.02124)

## TODOs

- [ ] Find $p^*$ and then add to figures.
- [ ] Do the actual writeup
- [ ] 1-2 more figures comparing local to global experimentation

## Short-Term Fix for `uv`

For whatever reason, `uv` will sometimes only install `demo` if `pyproject.toml` has been saved very recently. When this is happening, I get `ModuleNotFoundError`s when importing `demo` inside scripts called by `uv run`. I have spent a lot of time digging into this and have not been able to find the root cause. The intermittency makes it very challenging to diagnose.

The short-term solution is to to use the following:

```bash
# Before using console
touch pyproject.toml && uv sync
# Instead of `uv run`
touch pyproject.toml && uv run
```

`scripts/diagnose_modulenotfound_error.py` checks that this fixes the `ModuleNotFoundError`.

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