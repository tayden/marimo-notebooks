# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "marimo>=0.23.4",
#     "numpy",
#     "plotly",
#     "polars",
#     "utide",
# ]
# ///

import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")

with app.setup:
    from datetime import timedelta

    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    import polars as pl
    import utide
    from plotly.subplots import make_subplots


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Pruth Harbour Tidal Analysis

    Harmonic analysis of tide gauge observations from Pruth Harbour (51.654°N) using `utide`. We decompose the water level record into tidal constituents, then validate the fitted model on a held-out year.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Load data

    The dataset contains 15-minute tide gauge observations. Timestamps are stored in UTC and shifted to local time (UTC+8).
    """)
    return


@app.cell
def _():
    file = mo.notebook_location() / "public" / "PruthTides2018_2025_15min.csv"
    
    df_raw = pl.read_csv(str(file))
    df_raw = df_raw.with_columns(
        pl.col("time").str.to_datetime() + timedelta(hours=8)
    )
    df_raw
    return (df_raw,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Train / validation split

    We hold out 2025 observations as the validation set. The model is fit on all earlier years (2018–2024), then evaluated on the unseen 2025 data. This tests whether the fitted constituents generalise beyond the training period rather than merely interpolating it.
    """)
    return


@app.cell
def _(df_raw):
    df_split = df_raw.with_columns(
        (pl.col("time").dt.year() >= 2025).alias("is_val")
    )
    df, df_val = df_split.partition_by("is_val")

    with mo.redirect_stdout():
        print(len(df), "train records")
        print(len(df_val), "val records")

    mo.vstack([df.head(3), df_val.head(3)])
    return df, df_val


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Fit tidal constituents with `utide`

    `utide.solve` decomposes the water level into a sum of sinusoids at astronomical frequencies. Each constituent has a fixed angular frequency set by celestial mechanics; the fit determines location-specific amplitudes and phases:

    $$h(t) = h_0 + \sum_i A_i \cos(\omega_i t - \phi_i)$$

    We use ordinary least squares (`method="ols"`) with Monte Carlo confidence intervals and a Rayleigh criterion of 0.95 to avoid resolving frequency pairs that are too close to separate given the record length.
    """)
    return


@app.cell
def _(df):
    coefs = utide.solve(
        df["time"].to_numpy(),
        df["elevation"].to_numpy(),
        lat=51.654,
        conf_int="MC",
        method="ols",
        Rayleigh_min=0.95,
    )
    return (coefs,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Tidal constituents

    Each constituent is characterised by its amplitude $A$, phase $\phi$, and signal-to-noise ratio $\text{SNR} = A / \sigma_A$. Larger amplitudes drive more of the observed tidal range. SNR values below ~2 mean the amplitude is indistinguishable from noise — either the constituent is not significant at this location, or the record is too short to resolve it from a neighbouring frequency.
    """)
    return


@app.cell
def _(coefs):
    constituents_df = pl.DataFrame(
        {
            "constituent": coefs.name,
            "amplitude": coefs.A,
            "phase": coefs.g,
            "snr": coefs.A / coefs.A_ci,
        }
    ).sort("amplitude", descending=True)

    constituents_df
    return (constituents_df,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Poorly resolved constituents with SNR < 2 contribute little predictive value and can be excluded without meaningfully affecting reconstruction accuracy:
    """)
    return


@app.cell
def _(constituents_df):
    constituents_df.filter(pl.col("snr") < 2)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Validation

    We reconstruct tide heights over the held-out 2025 period and compare against observations. Residuals capture non-tidal variability — storm surge, atmospheric pressure, wind setup — that harmonic analysis cannot model. An $R^2$ of 0.9+ typically indicates well-characterised tidal constituents.

    - **RMSE**: $\sqrt{\frac{1}{n}\sum(h_\text{pred} - h_\text{obs})^2}$ — penalises large errors more heavily
    - **MAE**: $\frac{1}{n}\sum|h_\text{pred} - h_\text{obs}|$ — average error magnitude
    - **$R^2$**: $1 - \frac{\sum(h_\text{obs} - h_\text{pred})^2}{\sum(h_\text{obs} - \bar{h}_\text{obs})^2}$ — fraction of variance explained
    """)
    return


@app.cell
def _(coefs, df_val):
    test_time = df_val["time"].to_numpy()
    test_elevation = df_val["elevation"].to_numpy()

    reconstruction = utide.reconstruct(test_time, coefs)
    residuals = test_elevation - reconstruction.h

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Observed vs Fitted", "Residuals"),
    )

    fig.add_trace(
        go.Scatter(x=test_time, y=test_elevation, name="observed", opacity=0.7),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=test_time, y=reconstruction.h, name="fitted", opacity=0.7),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=test_time, y=residuals, name="residual", showlegend=False),
        row=2, col=1,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=2, col=1)

    fig.update_yaxes(title_text="elevation (m)", row=1, col=1)
    fig.update_yaxes(title_text="residual (m)", row=2, col=1)
    fig.update_xaxes(title_text="time", row=2, col=1)
    fig.update_layout(height=500, legend=dict(orientation="h", yanchor="bottom", y=1.02))

    fig
    return reconstruction, test_elevation, test_time


@app.cell
def _(reconstruction, test_elevation):
    residuals_1 = test_elevation - reconstruction.h
    rmse = np.sqrt(np.mean(residuals_1 ** 2))
    mae = np.mean(np.abs(residuals_1))
    ss_res = np.sum(residuals_1 ** 2)
    ss_tot = np.sum((test_elevation - np.mean(test_elevation)) ** 2)
    r_squared = 1 - ss_res / ss_tot

    with mo.redirect_stdout():
        print(f"RMSE: {rmse:.3f} m")
        print(f"MAE:  {mae:.3f} m")
        print(f"R²:   {r_squared:.3f}")
    return


if __name__ == "__main__":
    app.run()
