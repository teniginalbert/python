import numpy as np 
import matplotlib.pyplot as plt 
#%matplotlib inline
import pandas as pd 
import mglearn
from IPython.display import display
plt.rc('font', family = 'Verdana')

x = np.linspace(-10, 10, 100)
y = np.sin(x)
plt.plot(x, y, marker = "x")