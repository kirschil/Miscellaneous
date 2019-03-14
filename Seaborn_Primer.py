# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 11:09:11 2019

@author: kirschil
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="darkgrid")
tips = sns.load_dataset("tips")

sns.scatterplot(x="total_bill", y="tip", hue="sex", style="time", data=tips)

df = pd.DataFrame(dict(time=np.arange(500), value=np.random.randn(500).cumsum()))

plota = sns.relplot(kind="line", x="time", y="value", data=df)
plt.figtext(0.45, 0.96, "Value", fontsize='large', color='blue', ha ='right')
plt.figtext(0.56, 0.96, "Time", fontsize='large', color='orange', ha ='left')
plt.figtext(0.45, 0.96, ' Over ', fontsize='large', color='black', ha ='left')
plt.axhspan(-20, 25, 0.045, .115, color="red", alpha=0.5)
plt.axhspan(-20, 25, .257, .37, color="red", alpha=0.5)
plota.set_xlabels("Time")
plota.set_ylabels("Value")

#plt.title("Value Over Time")
