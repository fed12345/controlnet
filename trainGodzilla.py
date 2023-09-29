import torch
import stable_baselines3
import sys
import numpy as np
from quadcopter_env import Quadcopter3DGates
import os
# reload sympy
from sympy import *

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
taup = 0.14
tauq = 0.14
taur = 0.14
tauT = 0.14

p_min = -6.0
p_max = 6.0
q_min = -6.0
q_max = 6.0
r_min = -2.0
r_max = 2.0
T_min = 0.0
T_max = 16.0

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

axs, ays, azs = R*Matrix([0,0,x])

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

num = 10
env = Quadcopter3DGates(num_envs=num, gates_pos=gate_pos, gate_yaw=gate_yaw, start_pos=start_pos, pause_if_collision=False, f_func=f_func)

# Run a random agent
env.reset()

done = False
def run():
    global done
    action = np.random.uniform(-1,1, size=(num,4))
    # action[:,3] = 1
    state, reward, done, _ = env.step(action)
    # print(state[0][-4:])
    # print(env.disturbances[0])
    if reward[0] > 1:
        print("reward:", reward)
    return env.render()

#animation.view(run, gate_pos=gate_pos, gate_yaw=gate_yaw) #, record_steps=1000, show_window=True)

import os
from stable_baselines3 import PPO
from datetime import datetime
from stable_baselines3.common.vec_env import VecMonitor
import importlib
from quadcopter_animation import animation

models_dir = 'models/PPO_INDI'
log_dir = 'logs/PPO_INDI'
video_log_dir = 'videos/PPO_INDI'

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
if not os.path.exists(video_log_dir):
    os.makedirs(video_log_dir)

# Date and time string for unique folder names
datetime_str = datetime.now().strftime("%Y%m%d-%H%M%S")

# Create the environment
env = Quadcopter3DGates(num_envs=100, gates_pos=gate_pos, gate_yaw=gate_yaw, start_pos=start_pos, gates_ahead=1, f_func=f_func)
test_env = Quadcopter3DGates(num_envs=10, gates_pos=gate_pos, gate_yaw=gate_yaw, start_pos=start_pos, gates_ahead=1, pause_if_collision=True, f_func=f_func)

# Wrap the environment in a Monitor wrapper
env = VecMonitor(env)

# MODEL DEFINITION
policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=[dict(pi=[120,120,120], vf=[120,120,120])], log_std_init = 0)
model = PPO(
    "MlpPolicy",
    env,
    policy_kwargs=policy_kwargs,
    verbose=0,
    tensorboard_log=log_dir,
    n_steps=1000,
    batch_size=5000,
    n_epochs=10,
    gamma=0.999
)

print(model.policy)
print(model.num_timesteps)

def animate_policy(model, env, deterministic=False, **kwargs):
    env.reset()
    def run():
        actions, _ = model.predict(env.states, deterministic=deterministic)
        states, rewards, dones, infos = env.step(actions)
        return env.render()
    #animation.view(run, gate_pos=env.gate_pos, gate_yaw=env.gate_yaw, **kwargs)

# animate untrained policy (use this to set the recording camera position)
animate_policy(model, test_env)

def train(model, test_env, log_name, n=10000000000):
    # save every 10 policy rollouts
    TIMESTEPS = model.n_steps*env.num_envs*10
    for i in range(0,n):
        
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=log_name)
        time_steps = model.num_timesteps
        # save model
        model.save(models_dir + '/' + log_name + '/' + str(time_steps))
        # save policy animation
        animate_policy(
            model,
            test_env,
            record_steps=1000,
            record_file=video_log_dir + '/' + log_name + '/' + str(time_steps) + '.mp4',
            show_window=False
        )
        all_items = os.listdir(models_dir + '/' + log_name)
        if len(all_items) > 50:
            break


train(model, test_env, 'INDI_TEST2_with_rate_penalty')