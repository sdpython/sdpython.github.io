import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import identity, array

M = identity(3)
W = array([[0.5, 0.5, 0], [0, 0, 1]]).T
H = array([[1, 1, 0], [0.0, 0.0, 1.0]])
wh = W @ H

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(M[:,0], M[:,1], M[:,2], c='b', marker='o', s=600, alpha=0.5)
ax.scatter(wh[:,0], wh[:,1], wh[:,2], c='r', marker='^')
plt.show()