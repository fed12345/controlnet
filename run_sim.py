import torch
import stable_baselines3
import sys
import numpy as np
from quadcopter_env import Quadcopter3DGates
import os
from stable_baselines3 import PPO
from datetime import datetime
from stable_baselines3.common.vec_env import VecMonitor
import importlib
# reload sympy
from sympy import *

path = "models/776000000_fast_yaw.zip"

print("python version:", sys.version)
print("stable_baselines3 version:", stable_baselines3.__version__)
print("torch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
print("cudnn version:", torch.backends.cudnn.version())

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

# set torch default device
torch.set_default_device(device)



# Equations of motion 3D quadcopter + INDI inner loop

state = symbols('x y z v_x v_y v_z phi theta psi p q r T_norm')
x,y,z,vx,vy,vz,phi,theta,psi,p,q,r,T_norm = state
# commands normalized to [-1,1]
control = symbols('p_cmd,q_cmd,r_cmd,T_cmd')
p_cmd,q_cmd,r_cmd,T_cmd = control

# Physics
g = 9.81

# INDI params
taup =  0.038205684394448125
tauq =  0.036791804
taur = 1.1111354055
tauT = 0.05654755

p_min = -0.6
p_max = 0.6
q_min = -0.6
q_max = 0.6
r_min = -.6
r_max = .6
T_min = 0.0
T_max = 9.81*1.1
# Drag model
k_x = 0.33915248
k_y = 0.4314916

# Rotation matrix 
Rx = Matrix([[1, 0, 0], [0, cos(phi), -sin(phi)], [0, sin(phi), cos(phi)]])
Ry = Matrix([[cos(theta), 0, sin(theta)], [0, 1, 0], [-sin(theta), 0, cos(theta)]])
Rz = Matrix([[cos(psi), -sin(psi), 0], [sin(psi), cos(psi), 0], [0, 0, 1]])
R = Rz*Ry*Rx

# Body velocity
vbx, vby, vbz = R.T@Matrix([vx,vy,vz])

# Drag
Dx = -k_x*vbx
Dy = -k_y*vby

# Dynamics
d_x = vx
d_y = vy
d_z = vz

T = (T_norm+1)/2*(T_max-T_min)+T_min
d_vx, d_vy, d_vz = Matrix([0,0,g]) + R@Matrix([Dx, Dy,-T])

d_phi   = p + q*sin(phi)*tan(theta) + r*cos(phi)*tan(theta)
d_theta = q*cos(phi) - r*sin(phi)
d_psi   = q*sin(phi)/cos(theta) + r*cos(phi)/cos(theta)

d_p      = ((p_cmd+1)/2*(p_max-p_min)+p_min - p)/taup
d_q      = ((q_cmd+1)/2*(q_max-q_min)+q_min - q)/tauq
d_r      = ((r_cmd+1)/2*(r_max-r_min)+r_min - r)/taur
d_T_norm = (T_cmd - T_norm)/tauT

# State space model
f = [d_x, d_y, d_z, d_vx, d_vy, d_vz, d_phi, d_theta, d_psi, d_p, d_q, d_r, d_T_norm]

# lambdify
f_func = lambdify((Array(state), Array(control)), Array(f), 'numpy')

import importlib
from quadcopter_animation import animation
importlib.reload(animation)

# Define the race track
# gate_pos = np.array([
#     [-1.5,-2,-1.5],
#     [1.5,2,-1.5],
#     [1.5,-2,-2.5],
#     [-1.5,2,-1.5]
# ]*2)

gate_pos = np.array([
    [ 2,-1.5,-1.5],
    [ 2, 1.5,-1.5],
    [-2, 1.5,-1.5],
    [-2,-1.5,-1.5]
]*2)

# gate_yaw = np.array([
#     0,
#     0,
#     np.pi,
#     np.pi
# ]*2)
gate_yaw = np.array([
    np.pi/4,
    3*np.pi/4,
    5*np.pi/4,
    7*np.pi/4
]*2)


# start_pos = gate_pos[0] - np.array([2,0,0])
start_pos = gate_pos[3] #- np.array([2,0,0])

def animate_policy(model, env, deterministic=False, **kwargs):
    env.reset()
    def run():
        actions, _ = model.predict(env.states, deterministic=deterministic)
        states, rewards, dones, infos = env.step(actions)
        return env.render()
    animation.view(run, gate_pos=env.gate_pos, gate_yaw=env.gate_yaw, **kwargs)


#animation.view(run, gate_pos=gate_pos, gate_yaw=gate_yaw) #, record_steps=1000, show_window=True)

test_env = Quadcopter3DGates(num_envs=1, gates_pos=gate_pos, gate_yaw=gate_yaw, start_pos=start_pos,f_func=f_func , gates_ahead=0, pause_if_collision=True, record_sim = True)

model = PPO.load(path)
animate_policy(model, test_env, deterministic=False)
