
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
    # fliter real from when nn
    global row
    row = 0
    def run():
        #print(row)
        global row
        entry = dict(zip(["x", "y", "z", "phi", "theta", "psi"],zip(sim_df.iloc[row], real_df.iloc[row])))
        row += 1
        return {**entry}
    animation.view(run, gate_pos=gate_pos, gate_yaw=gate_yaw,fps=1000, **kwargs)

real_csv = "/Users/Federico/Desktop/Thesis/code/paper_figures/gcnet_data/data/example_8.csv"
sim_csv = "/Users/Federico/Desktop/Thesis/code/paper_figures/gcnet_data/sim/sim.csv"
animate_policy(sim_csv, real_csv, deterministic=False)