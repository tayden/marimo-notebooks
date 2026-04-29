# marimo Notebooks

Interactive data science notebooks built with [marimo](https://marimo.io), deployed to GitHub Pages as WebAssembly apps.

**Live site: [tayden.github.io/marimo-notebooks](https://tayden.github.io/marimo-notebooks/)**

## Notebooks

### Apps (`apps/` — read-only, interactive)

- `apps/chs_reverse_engineering.py`: Tidal analysis and prediction using CHS water level data, with harmonic decomposition via utide
- `apps/icefield_projector_calibration_explainer.py`: Interactive explainer for the pinhole camera model and projector calibration system used in the Icefield Projector

### Notebooks (`notebooks/` — editable)

- `notebooks/pruth-tides.py`: Tidal analysis for the Pruth Bay area using historical water level data (2018–2025)

## Local development

Run a notebook locally:

```bash
uv run marimo edit notebooks/pruth-tides.py
```

Build the static site:

```bash
uv run .github/scripts/build.py
```

Serve the built site:

```bash
python -m http.server -d _site
```

## Deployment

Pushing to `main` triggers a GitHub Actions workflow that exports all notebooks to WebAssembly and deploys to GitHub Pages. No manual steps required.

To add a new notebook:

1. Drop a `.py` marimo file into `notebooks/` (editable mode) or `apps/` (read-only mode)
2. Add any data or assets to the corresponding `public/` subdirectory
3. Push to `main`

## Including data or assets

Place data files alongside notebooks in a `public/` subdirectory:

```python
import polars as pl
df = pl.read_csv(mo.notebook_location() / "public" / "data.csv")
```