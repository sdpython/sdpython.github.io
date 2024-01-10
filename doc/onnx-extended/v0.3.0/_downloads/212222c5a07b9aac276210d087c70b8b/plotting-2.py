import pandas
from onnx_extended.plotting.data import vhistograms_data
from onnx_extended.plotting.benchmark import vhistograms

df = pandas.DataFrame(vhistograms_data())
vhistograms(df)