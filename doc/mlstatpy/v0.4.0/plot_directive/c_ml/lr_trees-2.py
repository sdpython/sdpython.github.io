import matplotlib.pyplot as plt
from mlstatpy.ml.logreg import criteria2, random_set_1d, plot_ds

X1, y1 = random_set_1d(1000, 2)
X2, y2 = random_set_1d(1000, 3)
X3, y3 = random_set_1d(1000, 4)
df1 = criteria2(X1, y1)
df2 = criteria2(X2, y2)
df3 = criteria2(X3, y3)
print(df3)

fig, ax = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
plot_ds(X1, y1, ax=ax[0], title="easy")
plot_ds(X2, y2, ax=ax[1], title="difficult")
plot_ds(X3, y3, ax=ax[2], title="more difficult")
df1.plot(x='X', y=['LL', 'LL-10', 'LL-100'], ax=ax[0], lw=5.)
df2.plot(x='X', y=['LL', 'LL-10', 'LL-100'], ax=ax[1], lw=5.)
df3.plot(x='X', y=['LL', 'LL-10', 'LL-100'], ax=ax[2], lw=5.)
plt.show()