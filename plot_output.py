import h5py
import sys
import numpy as np
from matplotlib import pyplot as plt

fname = sys.argv[1]

d = h5py.File(fname, "r")
data = np.array(d["band"][:,:,:]).squeeze(-1).T

plt.matshow(data)
plt.show()
