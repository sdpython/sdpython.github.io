import matplotlib.pyplot as plt
from pyquickhelper.pycode.profiling import profile
from pyquickhelper.texthelper import compare_module_version

def fctm():
    return compare_module_version('0.20.4', '0.22.dev0')

pr, df = profile(lambda: [fctm() for i in range(0, 1000)], as_df=True)
ax = df[['namefct', 'cum_tall']].head(n=15).set_index(
    'namefct').plot(kind='bar', figsize=(8, 3), rot=30)
ax.set_title("example of a graph")
for la in ax.get_xticklabels():
    la.set_horizontalalignment('right');
plt.show()