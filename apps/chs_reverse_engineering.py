# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "aiodns==4.0.0",
#     "aiohttp==3.13.5",
#     "marimo>=0.23.4",
#     "numpy==2.3.5",
#     "pandas==2.3.3",
#     "plotly==6.5.0",
#     "polars==1.36.1",
#     "utide==0.3.1",
# ]
# ///

import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")

with app.setup:
    import asyncio
    from datetime import datetime, date, timedelta
    from itertools import pairwise

    import marimo as mo
    import aiohttp
    import utide
    import numpy as np
    import polars as pl
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Reverse Engineering CHS Tide Predictions

    The Canadian Hydrographic Service (CHS) publishes tide predictions for hundreds of stations along Canada's coastlines via their IWLS API. Under the hood, those predictions come from **tidal harmonic analysis** — a technique that decomposes observed water levels into a sum of sinusoidal waves, each driven by a specific astronomical cycle.

    This notebook reverse-engineers that process. We download historical observations directly from CHS, fit our own harmonic model using [`utide`](https://github.com/wesleybowman/UTide), and use the fitted model to reconstruct tide predictions — bypassing CHS's rate-limited prediction API entirely.
    """)
    return


@app.function
async def get_stations(region_code: str = "PAC"):
    async with aiohttp.ClientSession() as session:
        async with session.get(
            "https://api.iwls-sine.azure.cloud-nuage.dfo-mpo.gc.ca/api/v1/stations",
            params={"chs-region-code": region_code},
        ) as response:
            if response.ok:
                return await response.json()
            raise RuntimeError("Could not fetch stations")


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Step 1 — Choose a Tide Station

    CHS maintains a network of tide gauges along Canada's Pacific, Arctic, and Atlantic coasts. Each gauge continuously records water level. Select one below — the station's latitude will be used to account for the nodal correction, a long-period modulation of tidal amplitudes driven by the 18.6-year cycle of the lunar nodes.
    """)
    return


@app.cell(hide_code=True)
async def _():
    stations = await get_stations()
    stations.sort(key=lambda s: s["officialName"])

    dropdown = mo.ui.dropdown(
        options=dict((s["officialName"], s) for s in stations),
        value="Adams Harbour",
        label="Choose a tide station",
        searchable=True,
    )

    dropdown
    return (dropdown,)


@app.cell(hide_code=True)
def _(dropdown):
    selected_station = dropdown.value
    return (selected_station,)


@app.function
async def get_tide_data(
        station_id: str,
        start_date: datetime | date,
        end_date: datetime | date,
        time_series_code: str = "wlp",
):
    date_range = pd.date_range(start=start_date, end=end_date, freq='MS')
    dates = [d.strftime("%Y-%m-%dT%H:%M:%SZ") for d in date_range]

    result = []

    async with aiohttp.ClientSession() as session:
        for from_, to_ in mo.status.progress_bar(list(pairwise(dates))):
            await asyncio.sleep(2)  # CHS rate limit: ~3 req/s, 30 req/min
            async with session.get(
                f"https://api.iwls-sine.azure.cloud-nuage.dfo-mpo.gc.ca/api/v1/stations/{station_id}/data",
                params={
                    "time-series-code": time_series_code,
                    "from": from_,
                    "to": to_,
                    "resolution": "FIFTEEN_MINUTES",
                },
            ) as response:
                if response.ok:
                    d = await response.json()
                    result.extend(d)
                else:
                    raise RuntimeError(f"Could not fetch station {station_id} for dates {from_} -> {to_}")

    return result


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Step 2 — Download Training Observations

    To fit a harmonic model, we need a sufficiently long record of observed water levels. Roughly a year of 15-minute data is ideal — it gives the solver enough cycles to cleanly resolve constituents that are close in frequency, including annual and semi-annual signals that take many months to separate from the dominant lunar tides.

    The CHS API imposes rate limits (~3 requests/second), so larger date ranges take a few minutes to fetch. Click to begin.
    """)
    return


@app.cell(hide_code=True)
def _():
    date_range = mo.ui.date_range(
        label="Fit tide model to 15min tides between",
        value=(date(2024, 1, 1), date(2025, 1, 1)),
    )
    date_range
    return (date_range,)


@app.cell(hide_code=True)
def _():
    should_fetch_tides_btn = mo.ui.run_button(label="Fetch training data")
    should_fetch_tides_btn
    return (should_fetch_tides_btn,)


@app.cell(hide_code=True)
async def _(date_range, selected_station, should_fetch_tides_btn):
    mo.stop(not should_fetch_tides_btn.value)

    with mo.redirect_stdout():
        print("Downloading observations...")

    tide_data = await get_tide_data(
        selected_station["id"],
        date_range.value[0],
        date_range.value[1],
    )
    return (tide_data,)


@app.cell(hide_code=True)
def _(tide_data):
    df = pl.DataFrame(tide_data)
    df = df.select(
        pl.col("eventDate").str.to_datetime(format="%Y-%m-%dT%H:%M:%SZ").alias("time"),
        pl.col("value").alias("elevation"),
    )
    return (df,)


@app.cell(hide_code=True)
def _(df):
    fig_raw = go.Figure()
    fig_raw.add_trace(go.Scatter(
        x=df["time"].to_numpy(),
        y=df["elevation"].to_numpy(),
        mode="lines",
        line=dict(width=0.6, color="steelblue"),
        name="observed",
    ))
    fig_raw.update_layout(
        title="Observed Water Levels — Training Data",
        xaxis_title="Time",
        yaxis_title="Elevation (m)",
        height=280,
        margin=dict(t=40, b=40),
    )
    fig_raw
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Step 3 — Fit the Harmonic Model

    The tide at any location can be written as a sum of sinusoidal waves:

    $$h(t) = h_0 + \sum_i A_i \cos(\omega_i t - \phi_i)$$

    Each term is a **tidal constituent** — a wave with:
    - a **fixed angular frequency** $\omega_i$ determined by celestial mechanics (the Moon's orbit, Earth's rotation, the Sun's apparent motion, and their combinations)
    - a **location-specific amplitude** $A_i$ and **phase offset** $\phi_i$ fitted from observations

    Because the frequencies are known from astronomy, the fitting problem reduces to ordinary least squares: given thousands of observed $(t, h)$ pairs, solve for the amplitudes and phases of ~35–70 constituents simultaneously.

    `utide` handles this. We pass it our time series and station latitude, and it returns the full set of constituent parameters:
    """)
    return


@app.cell(hide_code=True)
def _(df, selected_station):
    coef = utide.solve(
        df["time"].to_numpy(),
        df["elevation"].to_numpy(),
        lat=selected_station["latitude"],
        nodal=True,
        trend=True,
        method="ols",
        conf_int="MC",
        Rayleigh_min=0.95,
    )

    mo.show_code()
    return (coef,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Tidal Constituents

    The chart below shows the top 15 constituents by amplitude. Blue bars have SNR ≥ 2 (well-resolved); grey bars are near the noise floor. The **SNR** is $A / \sigma_A$ — how many standard deviations the fitted amplitude sits above zero.

    Common constituents to look for:

    | Name | Period | Origin |
    |------|--------|--------|
    | **M2** | 12.42 h | Principal lunar semidiurnal — dominant almost everywhere |
    | **S2** | 12.00 h | Principal solar semidiurnal |
    | **N2** | 12.66 h | Larger lunar elliptic semidiurnal |
    | **K1** | 23.93 h | Luni-solar diurnal |
    | **O1** | 25.82 h | Principal lunar diurnal |
    | **Sa** | 365.25 d | Solar annual — captures seasonal mean sea level variation |

    Locations with mixed or diurnal tides (e.g. much of BC's coast) will show large K1 and O1 relative to M2.
    """)
    return


@app.cell(hide_code=True)
def _(coef):
    constituents_df = pd.DataFrame(
        {
            "constituent": coef.name,
            "amplitude": coef.A,
            "phase": coef.g,
            "snr": coef.A / coef.A_ci,
        }
    ).sort_values("amplitude", ascending=False)

    top_n = constituents_df.head(15)

    fig_constituents = go.Figure(go.Bar(
        x=top_n["constituent"],
        y=top_n["amplitude"],
        text=[f"SNR {v:.1f}" for v in top_n["snr"]],
        textposition="outside",
        marker_color=["steelblue" if snr >= 2 else "#c8d6e0" for snr in top_n["snr"]],
    ))
    fig_constituents.update_layout(
        title="Top 15 Tidal Constituents by Amplitude",
        xaxis_title="Constituent",
        yaxis_title="Amplitude (m)",
        height=350,
        margin=dict(t=40, b=40),
    )
    fig_constituents
    return (constituents_df,)


@app.cell(hide_code=True)
def _(constituents_df):
    constituents_df
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Step 4 — Validate Against Held-Out Observations

    A good harmonic model should generalize beyond the data it was trained on. To test this, we fetch independent tide observations from a *different* time period and compare the model's predictions to them.

    Select a date range that does **not** overlap with the training window above — this is a true out-of-sample test. The residual (observed minus fitted) reveals what the harmonic model cannot capture: storm surge, wind setup, atmospheric pressure anomalies, and other non-tidal signals.
    """)
    return


@app.cell(hide_code=True)
def _():
    test_date_range = mo.ui.date_range(
        label="Validation data range",
        value=(date(2020, 1, 1), date(2020, 6, 1)),
    )
    test_date_range
    return (test_date_range,)


@app.cell(hide_code=True)
def _():
    should_fetch_test_tides_btn = mo.ui.run_button(label="Fetch validation data")
    should_fetch_test_tides_btn
    return (should_fetch_test_tides_btn,)


@app.cell(hide_code=True)
async def _(selected_station, should_fetch_test_tides_btn, test_date_range):
    mo.stop(not should_fetch_test_tides_btn.value)

    with mo.redirect_stdout():
        print("Downloading validation observations...")

    test_tide_data = await get_tide_data(
        selected_station["id"],
        test_date_range.value[0],
        test_date_range.value[1],
    )
    test_df = pl.DataFrame(test_tide_data)
    test_df = test_df.select(
        pl.col("eventDate").str.to_datetime(format="%Y-%m-%dT%H:%M:%SZ").alias("time"),
        pl.col("value").alias("elevation"),
    )
    return (test_df,)


@app.cell(hide_code=True)
def _(test_df):
    test_time = test_df["time"].to_numpy()
    test_elevation = test_df["elevation"].to_numpy()
    return test_elevation, test_time


@app.cell(hide_code=True)
def _(coef, test_elevation, test_time):
    reconstruction = utide.reconstruct(test_time, coef)
    residuals = test_elevation - reconstruction.h

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Observed vs Model", "Residuals (non-tidal signal)"),
    )

    fig.add_trace(
        go.Scatter(x=test_time, y=test_elevation, name="observed", opacity=0.7, line=dict(width=0.8)),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=test_time, y=reconstruction.h, name="model", opacity=0.85, line=dict(width=1.2)),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=test_time, y=residuals, name="residual", showlegend=False, line=dict(width=0.7, color="grey")),
        row=2, col=1,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.3, row=2, col=1)

    fig.update_yaxes(title_text="elevation (m)", row=1, col=1)
    fig.update_yaxes(title_text="residual (m)", row=2, col=1)
    fig.update_xaxes(title_text="time", row=2, col=1)
    fig.update_layout(height=520, legend=dict(orientation="h", yanchor="bottom", y=1.02))

    fig
    return (reconstruction,)


@app.cell(hide_code=True)
def _(reconstruction, test_elevation):
    residuals_v = test_elevation - reconstruction.h
    rmse = np.sqrt(np.mean(residuals_v ** 2))
    mae = np.mean(np.abs(residuals_v))
    ss_res = np.sum(residuals_v ** 2)
    ss_tot = np.sum((test_elevation - np.mean(test_elevation)) ** 2)
    r_squared = 1 - ss_res / ss_tot

    mo.md(f"""
    | Metric | Value | Interpretation |
    |--------|-------|----------------|
    | **RMSE** | {rmse:.3f} m | Root mean squared error — penalises large misses |
    | **MAE** | {mae:.3f} m | Average absolute error |
    | **R²** | {r_squared:.3f} | Fraction of variance explained by the tidal model |
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    An R² above 0.90 indicates the tidal constituents are well-characterised. Remaining residuals are real oceanographic signal — primarily meteorological (storm surge, inverse barometer effect, wind-driven setup) — that harmonic analysis cannot capture because it is not periodic.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Step 5 — Generate Predictions at Scale

    Here's the payoff. CHS's prediction API imposes strict rate limits (3 req/s, 30 req/min) and restricts the time window per request at higher resolutions. Fetching years of fine-resolution predictions directly from the API is slow and often impractical.

    With a fitted harmonic model, `utide.reconstruct` generates predictions entirely in Python — no network calls, no rate limits. The computation is simple matrix arithmetic over the constituent parameters. Two years of 5-minute predictions (~210,000 values) runs in under a second:
    """)
    return


@app.cell(hide_code=True)
def _(coef):
    from time import perf_counter

    def dt_range(start: datetime, end: datetime, delta: timedelta):
        cur = start
        while cur <= end:
            yield cur
            cur += delta

    t0 = perf_counter()
    times_to_fetch = list(dt_range(datetime(2023, 1, 1), datetime(2025, 1, 1), timedelta(minutes=5)))
    tides_five_minutes = utide.reconstruct(times_to_fetch, coef).h
    elapsed = perf_counter() - t0

    mo.md(f"""
    Generated **{len(tides_five_minutes):,}** tide predictions at 5-minute resolution (2 years) in **{elapsed:.2f} s** — no API calls required.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ---

    **The key insight**: tide physics doesn't change from year to year. The astronomical frequencies are fixed by celestial mechanics, and the local amplitude and phase parameters are stable on decadal timescales. A model fitted once on a few hundred megabytes of historical observations stays valid indefinitely — the one-time cost of downloading training data pays off in unlimited, instant, offline predictions.
    """)
    return


if __name__ == "__main__":
    app.run()
