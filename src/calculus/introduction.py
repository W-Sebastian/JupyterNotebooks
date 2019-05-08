import numpy as np
import plotly.graph_objs as go
from plotly.offline import iplot

from functools import reduce

def compound_interest(day):
    return 10 * np.power(1.1, day)

def find_sum_of_n_days(start, end):
    days = np.arange(start, end, 1)
    dailyPayments = [compound_interest(x) for x in days]
    return sum(dailyPayments)

def find_price_difference(day):
    return pow(1.1, day-1)

def plot_compound_interest():
    days = np.arange(1, 30, 1)
    dailyPayment = [compound_interest(x) for x in days]

    trace = go.Scatter(
        x = days, 
        y = dailyPayment, 
        mode = 'markers', 
        name = 'markers')

    layout = dict(
        title = "How prices increases when renting a car",
        xaxis = dict(title = "Days of Renting"),
        yaxis = dict(title = "Cost each Day (€)")
    )
    fig = dict(data = [trace], layout = layout)

    iplot(fig)

def plot_price_increase():
    days = np.arange(1, 30, 1)
    dailyPayment = [find_price_difference(x) for x in days]

    trace = go.Scatter(
        x = days, 
        y = dailyPayment, 
        mode = 'markers', 
        name = 'markers')

    layout = dict(
        title = "How prices increases from a day to another",
        xaxis = dict(title = "# day"),
        yaxis = dict(title = "Cost Increase (€)")
    )
    fig = dict(data = [trace], layout = layout)

    iplot(fig)

if __name__ == "__main__":
    
    pass