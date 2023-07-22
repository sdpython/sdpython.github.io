import matplotlib.pyplot as plt
import numpy
def softmax(x):
    return 1.0 / (1 + numpy.exp(-x))
def dsoftmax(x):
    t = numpy.exp(-x)
    return t / (1 + t)**2
x = numpy.arange(-10,10, 0.1)
y = softmax(x)
dy = dsoftmax(x)
fig, ax = plt.subplots(1,1)
ax.plot(x,y, label="softmax")
ax.plot(x,dy, label="dérivée")
ax.set_ylim([-0.1, 1.1])
ax.plot([-5, -5], [-0.1, 1.1], "r")
ax.plot([5, 5], [-0.1, 1.1], "r")
ax.legend(loc=2)
plt.show()