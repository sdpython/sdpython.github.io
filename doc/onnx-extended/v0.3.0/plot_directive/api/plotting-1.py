import pandas
from onnx_extended.plotting.data import hhistograms_data
from onnx_extended.plotting.benchmark import hhistograms

df = pandas.DataFrame(hhistograms_data())
hhistograms(df, keys=("input", "name"))