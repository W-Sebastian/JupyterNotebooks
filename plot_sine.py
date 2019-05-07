
#%%

import plotly.plotly as py
import plotly
import numpy as np

from plotly.offline import init_notebook_mode

def sine_plotter():
    data = [dict(
            visible = False,
            line=dict(color='#224400', width=6),
            name = '𝜈 = '+str(step),
            x = np.arange(0,10,0.01),
            y = np.sin(step*np.arange(0,10,0.01))) for step in np.arange(0,5,0.1)]
    data[10]['visible'] = True

    steps = []
    for i in range(len(data)):
        step = dict(
            method = 'restyle',
            args = ['visible', [False] * len(data)],
        )
        step['args'][1][i] = True # Toggle i'th trace to "visible"
        steps.append(step)

    sliders = [dict(
        active = 10,
        currentvalue = {"prefix": "Frequency: "},
        pad = {"t": 50},
        steps = steps
    )]

    layout = dict(sliders=sliders)
    fig = dict(data=data, layout=layout)

    plotly.offline.iplot(fig, filename='Sine Wave Slider')


#%%



