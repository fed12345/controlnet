
import numpy as np
import pandas as pd
import importlib
import sys
from quadcopter_animation import animation
importlib.reload(animation)



gate_pos = np.array([
    [ 2,-1.5,-1.5],
    [ 2, 1.5,-1.5],
    [-2, 1.5,-1.5],
    [-2,-1.5,-1.5]
]*2)

gate_yaw = np.array([
    np.pi/4,
    3*np.pi/4,
    5*np.pi/4,
    7*np.pi/4
]*2)
def interpolate_data(data):
    """
    Function to perform linear interpolation on a DataFrame, adding one interpolated
    data point between each pair of existing points.

    Parameters:
    data (DataFrame): The original DataFrame to be interpolated.

    Returns:
    DataFrame: A new DataFrame with interpolated data.
    """
    # Prepare a new index that doubles the number of points
    new_index = np.linspace(start=0, stop=len(data) - 1, num=len(data) * 3 - 1)

    # Create a new DataFrame to hold the interpolated data
    interpolated_data = pd.DataFrame(index=new_index)

    # Interpolate each column linearly
    for column in data.columns:
        interpolated_data[column] = np.interp(new_index, np.arange(len(data)), data[column])

    # Reset the index of the new DataFrame to make it look cleaner
    interpolated_data.reset_index(drop=True, inplace=True)

    return interpolated_data

global row
def animate_policy(sim_csv, real_csv, deterministic=False, **kwargs):
    # Load the data from the CSV , only select centrain entries (otX0,otY0)
    sim_df = pd.read_csv(sim_csv, usecols=["otX0", "otY0", "otZ0", "otRoll0", "otPitch0", "otYaw0"], sep=",")
    real_df = pd.read_csv(real_csv, usecols=["otX0", "otY0", "otZ0", "otRoll0", "otPitch0", "otYaw0", "n_out1"], sep=", ")
    real_df = real_df[real_df['n_out1'] != 0]
    real_df["otY0"] = -real_df["otY0"]
    real_df["otZ0"] = -real_df["otZ0"]
    real_df["otRoll0"] = real_df["otPitch0"]*np.pi/180
    real_df["otPitch0"] = real_df["otRoll0"]*np.pi/180
    real_df["otYaw0"] = real_df["otYaw0"]*np.pi/180
    real_df = real_df.drop(columns=['n_out1'])

    real_df = interpolate_data(real_df)
    sim_df = interpolate_data(sim_df)

    global row
    row = 0
    def run():
        #print(row)
        global row
        entry = dict(zip(["x", "y", "z", "phi", "theta", "psi"],zip(sim_df.iloc[row], real_df.iloc[row])))
        row += 1
        return {**entry}
    animation.view(run, gate_pos=gate_pos, gate_yaw=gate_yaw,fps=300, **kwargs)

real_csv = "/Users/Federico/Desktop/Thesis/code/paper_figures/gcnet_data/data/example_8.csv"
sim_csv = "/Users/Federico/Desktop/Thesis/code/paper_figures/gcnet_data/sim/sim.csv"
animate_policy(sim_csv, real_csv, deterministic=False)