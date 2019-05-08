import numpy as np
import plotly.graph_objs as go
from plotly.offline import iplot

def qf(t):
  r = -9.8*(t**2) + 20*t + 1
  return r

def plot_quadratic_f():
    timeSteps = np.linspace(0, 2.08965, 100)
    height = [qf(t) for t in timeSteps]

    trace = go.Scatter(
        x = timeSteps, 
        y = height, 
        mode = 'lines', 
        name = 'height_over_time')

    layout = go.Layout(
        xaxis = dict(
            title = 'Time (s)',
            tickmode = 'linear',
            ticks = 'outside',
            tick0 = 0,
            dtick = 0.1
        ),
        yaxis = dict(
            title = 'Height (m)',
            tickmode = 'linear',
            ticks = 'outside'
        )
    )

    fig = go.Figure(data=[trace], layout = layout)
    iplot(fig)