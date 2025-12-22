import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("log.csv")
x = df['timestep']
y = df['reward']
plt.plot(x,y)
plt.xlabel('Timestep')
plt.ylabel('Reward')
plt.title('Reward')
plt.grid(True)
plt.show()
"""import matplotlib.pyplot as plt
import numpy as np

# 1. Define the data points
# You can use lists or numpy arrays
x_values = np.array([1, 2, 3, 4, 5])
y_values = np.array([2, 3, 5, 4, 6])

# 2. Plot the line
# The plot() function draws a line by default
plt.plot(x_values, y_values)

# 3. Add labels and a title (optional but recommended)
plt.xlabel('X-axis Label')
plt.ylabel('Y-axis Label')
plt.title('My First Line Plot')

# 4. Add grid lines (optional)
plt.grid(True)

# 5. Display the plot
plt.show()"""

