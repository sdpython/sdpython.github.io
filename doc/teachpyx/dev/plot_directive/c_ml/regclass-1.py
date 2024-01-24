from teachpyx.datasets import load_wines_dataset
import matplotlib.pyplot as plt
plt.close('all')

df = load_wines_dataset()

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,4))
df.quality.hist(bins=18, ax=ax)
plt.title('Distribution des notes des vins')
plt.show()