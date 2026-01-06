import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("src/controller/controller/models/SAC/SAC_log.csv")
x = df['timestep']
y = df['reward']
plt.plot(x,y)
plt.xlabel('Timestep')
plt.ylabel('Reward')
plt.title('Reward')
plt.grid(True)
plt.show()
