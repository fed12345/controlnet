import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt

# Load the data from the CSV
data = pd.read_csv('crazyflie_dataset/thrust.csv')
time = data['timeTick'].values *10e-3
dt = time[1] - time[0]
#make time start at 0
time = time - time[0]
input_signal = data[' desThrustCmd'].values 
output_signal = data[' desAccelZ'].values

# Estimate the derivative of the output
dydt = np.gradient(output_signal, time)

# yk+1 = yk + ((k * input_signal - yk)/tau) * dt
# dymdt = (k * input_signal - y)/tau) 
# y0 = 0

def firstorderdelay(input_signal, k, tau):
    model_signal = np.zeros(len(input_signal))
    model_signal[0] = 0
    for i in range(1, len(input_signal)):
        model_signal[i] = model_signal[i-1] + ((k * input_signal[i-1] - model_signal[i-1])/tau)*dt
    return model_signal
def firstoderdelaynoscale(input_signal, tau):
    return firstorderdelay(input_signal, 1, tau)
# Curve fitting to estimate dydt
popt, _ = curve_fit(firstorderdelay, input_signal, output_signal)

K_estimated, tau_estimated = popt
print("Estimated K:", K_estimated)
print("Estimated time constant tau:", tau_estimated)



plt.figure(figsize=(10, 8))
plt.subplot(2, 1, 1)
plt.plot(time, input_signal*K_estimated, label='Input')
plt.plot(time, output_signal, label='Output')
#modelled signal
plt.plot(time, firstorderdelay(input_signal,K_estimated, tau_estimated), label='Model')
plt.legend()
plt.show()






