#!/usr/bin/env python
# coding: utf-8

# In[1]:


import polars as pl
from datetime import timedelta
import utide
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo
import numpy as np

pyo.init_notebook_mode()


# In[2]:


df = pl.read_csv("./PruthTides2018_2025_15min.csv")
df = df.with_columns(
    pl.col("time").str.to_datetime() + timedelta(hours=8) # UTC datetimes
)

# Split into train/validation sets by year
df = df.with_columns(
    (pl.col("time").dt.year() >= 2025).alias("is_val")
)
df, df_val = df.partition_by("is_val")

print(len(df), "train records")
print(len(df_val), "val records")

display(df.head(3))
display(df_val.head(3))


# In[3]:


coefs = utide.solve(
    df["time"].to_numpy(),
    df["elevation"].to_numpy(),
    lat=51.654,
    conf_int="MC",
    method="ols",
    Rayleigh_min=0.95
)


# In[4]:


constituents_df = pl.DataFrame(
    {
        "constituent": coefs.name,
        "amplitude": coefs.A,
        "phase": coefs.g,
        "snr": coefs.A / coefs.A_ci,  # signal-to-noise ratio
    }
).sort("amplitude", descending=True)

constituents_df.filter(pl.col("snr") < 2)


# # Test solution

# In[5]:


test_time = df_val["time"].to_numpy()
test_elevation = df_val["elevation"].to_numpy()

reconstruction = utide.reconstruct(test_time, coefs)
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


# In[6]:


residuals_1 = test_elevation - reconstruction.h
rmse = np.sqrt(np.mean(residuals_1 ** 2))
mae = np.mean(np.abs(residuals_1))
ss_res = np.sum(residuals_1 ** 2)
ss_tot = np.sum((test_elevation - np.mean(test_elevation)) ** 2)
r_squared = 1 - ss_res / ss_tot

print(f'RMSE: {rmse:.3f} m')
print(f'MAE: {mae:.3f} m')
print(f'R²: {r_squared:.3f}')


# In[ ]:




