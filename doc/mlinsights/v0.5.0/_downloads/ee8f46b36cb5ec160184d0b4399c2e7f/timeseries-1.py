import datetime
import matplotlib.pyplot as plt
from mlinsights.timeseries.datasets import artificial_data
from mlinsights.timeseries.agg import aggregate_timeseries
from mlinsights.timeseries.plotting import plot_week_timeseries

dt1 = datetime.datetime(2019, 8, 1)
dt2 = datetime.datetime(2019, 9, 1)
data = artificial_data(dt1, dt2, minutes=15)
print(data.head())

agg = aggregate_timeseries(data, per='week')
plot_week_timeseries(
    agg['weektime'], agg['y'], label="y",
    value2=agg['y']/2, label2="y/2", normalise=False)
plt.show()