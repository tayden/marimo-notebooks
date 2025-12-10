# /// script
# dependencies = [
#     "httpx==0.28.1",
#     "numpy==2.3.5",
#     "pandas==2.3.3",
#     "plotly==6.5.0",
#     "polars==1.36.1",
#     "utide==0.3.1",
# ]
# ///

import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")

with app.setup:
    from datetime import datetime, date, timedelta
    from itertools import pairwise
    from time import sleep

    import marimo as mo
    import httpx
    import utide
    import numpy as np
    import polars as pl
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Reverse Engineering CHS Tide Models with `utide`
    """)
    return


@app.function
def get_stations(region_code: str = "PAC"):
    stations = httpx.get(
        "https://api.iwls-sine.azure.cloud-nuage.dfo-mpo.gc.ca/api/v1/stations",
        params={
            "chs-region-code": region_code,
        }
    )
    if stations.is_success:
        return stations.json()

    raise ValueError("Something went wrong")


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Fetch station data to build our model

    Select a station and a date range to build the model on. Models built with about one year of data tend to perform well, but feel free to experiment with shorter or longer ranges as well.
    """)
    return


@app.cell
def _():
    stations = get_stations()
    stations.sort(key=lambda s: s["officialName"])

    dropdown = mo.ui.dropdown(
        options=dict((s["officialName"], s) for s in stations),
        value="Adams Harbour",
        label="Choose a tide station",
        searchable=True,
    )

    dropdown
    return (dropdown,)


@app.cell
def _(dropdown):
    selected_station = dropdown.value
    return (selected_station,)


@app.function
def get_tide_data(
        station_id: str,
        start_date: datetime | date,
        end_date: datetime | date,
        time_series_code: str = "wlp",
):
    date_range = pd.date_range(start=start_date, end=end_date, freq='MS')
    dates = [d.strftime("%Y-%m-%dT%H:%M:%SZ") for d in date_range]

    result = []

    for from_, to_ in mo.status.progress_bar(list(pairwise(dates))):
        sleep(2)  # Have to go slow to not hit the CHS api limits...
        res = httpx.get(
            f"https://api.iwls-sine.azure.cloud-nuage.dfo-mpo.gc.ca/api/v1/stations/{station_id}/data",
            params={
                "time-series-code": time_series_code,
                "from": from_,
                "to": to_,
                "resolution": "FIFTEEN_MINUTES"
            }
        )
        result.extend(res.json())

    return result


@app.cell
def _():
    date_range = mo.ui.date_range(label="Fit tide model to 15min tides between",
                                  value=(date(2024, 1, 1), date(2025, 1, 1)))
    date_range
    return (date_range,)


@app.cell
def _():
    should_fetch_tides_btn = mo.ui.run_button(label="Click to continue and fetch the tide data")
    should_fetch_tides_btn
    return (should_fetch_tides_btn,)


@app.cell
def _(date_range, selected_station, should_fetch_tides_btn):
    mo.stop(not should_fetch_tides_btn.value)

    with mo.redirect_stdout():
        print(f"Downloading data...")

    tide_data = get_tide_data(
        selected_station["id"],
        date_range.value[0],
        date_range.value[1],
    )
    return (tide_data,)


@app.cell
def _(tide_data):
    df = pl.DataFrame(tide_data)
    df = df.select(
        pl.col("eventDate").str.to_datetime(format="%Y-%m-%dT%H:%M:%SZ").alias("time"),
        pl.col("value").alias("elevation")
    )
    df
    return (df,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Now we solve for the tide constituent components using `utide`
    """)
    return


@app.cell
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
    Tidal constituents represent periodic gravitational forcing from the moon and sun. Each constituent has a fixed frequency determined by celestial mechanics — orbital periods, Earth's rotation, and their interactions. For example, **M2** (the dominant constituent in most locations) corresponds to the principal lunar semidiurnal cycle at 12.42 hours, while **S2** captures the solar semidiurnal cycle at exactly 12 hours.

    The amplitude $A$ and phase $\phi$ are location-specific parameters that describe how each astronomical forcing manifests locally. Coastal geometry, bathymetry, and resonance effects mean the same constituent can have vastly different amplitudes at different locations. The fitted model predicts tide height as:

    $$h(t) = h_0 + \sum_i A_i \cos(\omega_i t - \phi_i)$$

    where $h_0$ is mean sea level and $\omega_i$ is the angular frequency of each constituent.

    The signal-to-noise ratio ($\text{SNR} = A / \sigma_A$) indicates how well-resolved each constituent is given your observation record. Values below ~2 suggest the amplitude is not distinguishable from noise — typically because the record is too short to separate nearby frequencies, or that constituent simply isn't significant at this location.
    """)
    return


@app.cell
def _(coef):
    constituents_df = pd.DataFrame(
        {
            "constituent": coef.name,
            "amplitude": coef.A,
            "phase": coef.g,
            "snr": coef.A / coef.A_ci,  # signal-to-noise ratio
        }
    ).sort_values("amplitude", ascending=False)

    constituents_df
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    To validate the model, we compare predictions against observations from a different time period than was used for fitting. This tests whether the fitted constituents generalize rather than just interpolating the training data.

    We retrieve independent tide observations from CHS and compute:

    - **RMSE** (root mean squared error): $\sqrt{\frac{1}{n}\sum(h_{pred} - h_{obs})^2}$ — penalizes large errors more heavily
    - **MAE** (mean absolute error): $\frac{1}{n}\sum|h_{pred} - h_{obs}|$ — average magnitude of errors
    - **$R^2$** (coefficient of determination): $1 - \frac{\sum(h_{obs} - h_{pred})^2}{\sum(h_{obs} - \bar{h}_{obs})^2}$ — proportion of variance explained by the model

    Residuals will always contain some non-tidal signal (storm surge, atmospheric pressure, wind setup) that harmonic analysis cannot capture. An $R^2$ of 0.9+ typically indicates the tidal constituents are well-characterized.
    """)
    return


@app.cell
def _():
    test_date_range = mo.ui.date_range(label="Test tide data range",
                                       value=(date(2020, 1, 1), date(2020, 6, 1)))
    test_date_range
    return (test_date_range,)


@app.cell
def _():
    should_fetch_test_tides_btn = mo.ui.run_button(label="Click to continue and fetch the tide data")
    should_fetch_test_tides_btn
    return (should_fetch_test_tides_btn,)


@app.cell
def _(selected_station, should_fetch_test_tides_btn, test_date_range):
    mo.stop(not should_fetch_test_tides_btn.value)

    with mo.redirect_stdout():
        print("Pulling test data...")

    test_tide_data = get_tide_data(
        selected_station["id"],
        test_date_range.value[0],
        test_date_range.value[1],
    )
    test_df = pl.DataFrame(test_tide_data)
    test_df = test_df.select(
        pl.col("eventDate").str.to_datetime(format="%Y-%m-%dT%H:%M:%SZ").alias("time"),
        pl.col("value").alias("elevation")
    )
    return (test_df,)


@app.cell
def _(test_df):
    test_time = test_df["time"].to_numpy()
    test_elevation = test_df["elevation"].to_numpy()
    return test_elevation, test_time


@app.cell
def _(coef, test_elevation, test_time):
    reconstruction = utide.reconstruct(test_time, coef)
    residuals = test_elevation - reconstruction.h

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=('Observed vs Fitted', 'Residuals'),
    )

    fig.add_trace(
        go.Scatter(x=test_time, y=test_elevation, name='observed', opacity=0.7),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=test_time, y=reconstruction.h, name='fitted', opacity=0.7),
        row=1, col=1,
    )

    fig.add_trace(
        go.Scatter(x=test_time, y=residuals, name='residual', showlegend=False),
        row=2, col=1,
    )
    fig.add_hline(y=0, line_dash='dash', line_color='gray', opacity=0.5, row=2, col=1)

    fig.update_yaxes(title_text='elevation (m)', row=1, col=1)
    fig.update_yaxes(title_text='residual (m)', row=2, col=1)
    fig.update_xaxes(title_text='time', row=2, col=1)

    fig.update_layout(height=500, legend=dict(orientation='h', yanchor='bottom', y=1.02))

    fig
    return (reconstruction,)


@app.cell
def _(reconstruction, test_elevation):
    residuals_1 = test_elevation - reconstruction.h
    rmse = np.sqrt(np.mean(residuals_1 ** 2))
    mae = np.mean(np.abs(residuals_1))
    ss_res = np.sum(residuals_1 ** 2)
    ss_tot = np.sum((test_elevation - np.mean(test_elevation)) ** 2)
    r_squared = 1 - ss_res / ss_tot

    with mo.redirect_stdout():
        print(f'RMSE: {rmse:.3f} m')
        print(f'MAE: {mae:.3f} m')
        print(f'R²: {r_squared:.3f}')
    return


@app.cell
def _():
    mo.md(r"""
    ## But why?

    CHS has pretty strict rate limits on their API. They'll only let you make 3 requests per second and 30 requests per minute. Furthermore, the more temporal resolution you request, the further they restrict the time periods you can request data for.

    So, with this, we can request huge amounts of data on our own servers and it's relatively quick. The only thing to watch out for is running out of RAM since the utide library doesn't stream the results.
    """)
    return


@app.cell
def _(coef):
    from time import perf_counter

    # Now we can pull a lot more tide data quickly...
    def dt_range(start: datetime, end: datetime, delta: timedelta):
        cur = start
        while cur <= end:
            yield cur
            cur += delta

    start_time = perf_counter()

    times_to_fetch = list(dt_range(datetime(2023, 1, 1), datetime(2025, 1, 1), timedelta(minutes=5)))
    tides_five_minutes = utide.reconstruct(times_to_fetch, coef).h

    elapsed_time = perf_counter() - start_time

    with mo.redirect_stdout():
        print(f"Fetched {len(tides_five_minutes)} tide values in {elapsed_time:.2f}s")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
