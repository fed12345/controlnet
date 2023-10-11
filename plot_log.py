import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('crazyflie_dataset/thrust.csv')

# Plot a 4,1 subplot of the data
figure, axes = plt.subplots(4, 1, figsize=(10, 8))


axes[2].plot(data['timeTick'], data[' desYawRate'], label='Desidered')
axes[2].plot(data['timeTick'], data[' gyroz'], label='Response')
axes[2].legend()
axes[2].set_title('Yaw')

plt.show()
