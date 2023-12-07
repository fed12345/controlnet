import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt

# Load the data from the CSV
data_yaw = pd.read_csv('crazyflie_dataset/yaw.csv')
#select only first 10000 entries
data_yaw = data_yaw.iloc[800:1400]
time_yaw = data_yaw['timeTick'].values * 1e-3


data_thrust = pd.read_csv('crazyflie_dataset/thrust.csv')
#select only first 10000 entries
data_thrust = data_thrust.iloc[300:1000]
time_thrust = data_thrust['timeTick'].values * 1e-3

data = pd.read_csv('crazyflie_dataset/rollandpitch.csv')
#select only first 10000 entries
data = data.iloc[1400:2000]
time = data['timeTick'].values * 1e-3

dt = time_yaw[1] - time_yaw[0]

#make time start at 0
input_signal = data_thrust[' desThrustCmd'].values 
output_signal = data_thrust[' desAccelZ'].values

input_signal_roll = data[' desRollRate'].values
output_signal_roll = data[' gyrox'].values

input_signal_pitch = data[' desPitchRate'].values
output_signal_pitch = -data[' gyroy'].values

input_signal_yaw = data_yaw[' desYawRate'].values
output_signal_yaw = data_yaw[' gyroz'].values


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
#popt, _ = curve_fit(firstorderdelay, input_signal, output_signal)

# K_estimated, tau_estimated = popt
# print("Estimated K:", K_estimated)
# print("Estimated time constant tau:", tau_estimated)


taup =  0.038205684394448125
tauq =  0.036791804
taur = 1.1111354055
tauT = 0.05654755
K_estimated = 2.0392824564266964e-05

plt.figure(figsize=(6, 8))
plt.subplot(4, 1, 1)
plt.title("System Identification of the Crazyflie")
plt.plot(time_thrust, input_signal*K_estimated, color = "#00A6D6", label='Input')
plt.plot(time_thrust, output_signal, color = 'r', label='Output')
#modelled signal
plt.plot(time_thrust, firstorderdelay(input_signal,K_estimated, tauT), color = 'g',label='Model')
#label axis
plt.xlabel('Time [s]')
plt.ylabel('Thrust [Gs]')
plt.grid()
plt.legend()

plt.subplot(4, 1, 2)
plt.plot(time, input_signal_roll, color = "#00A6D6", label='Input')
plt.plot(time, output_signal_roll, color = 'r', label='Output')
#modelled signal
plt.plot(time, firstorderdelay(input_signal_roll,1, taup), color = 'g',label='Model')
#label axis
plt.ylim(-250, 250)
plt.xlabel('Time [s]')
plt.ylabel('Roll [deg/s]')
plt.grid()

plt.subplot(4, 1, 3)
plt.plot(time, input_signal_pitch, color = "#00A6D6", label='Input')
plt.plot(time, output_signal_pitch, color = 'r', label='Output')
#modelled signal
plt.plot(time, firstorderdelay(input_signal_pitch,1, tauq), color = 'g',label='Model')
#label axis
plt.ylim(-200, 200)
plt.xlabel('Time [s]')
plt.ylabel('Pitch [deg/s]')
plt.grid()

plt.subplot(4, 1, 4)
plt.plot(time_yaw, input_signal_yaw, color = "#00A6D6", label='Input')
plt.plot(time_yaw, output_signal_yaw, color = 'r', label='Output')
#modelled signal
plt.plot(time_yaw, firstorderdelay(input_signal_yaw,1, taur), color = 'g',label='Model')
#label axis
plt.xlabel('Time [s]')
plt.ylabel('Yaw [deg/s]')
plt.grid()
#
plt.savefig('system_identification.svg', format= 'svg', dpi=300, bbox_inches='tight')

plt.show()






