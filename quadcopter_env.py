# Efficient vectorized version of the environment
from gym import spaces
from stable_baselines3.common.vec_env import VecEnv
import numpy as np

class Quadcopter3DGates(VecEnv):
    def __init__(self,
                 num_envs,
                 gates_pos,
                 gate_yaw,
                 start_pos,
                 f_func,
                 gates_ahead=0,
                 pause_if_collision=False,
                 
                 ):
        
        # Define the race track
        self.start_pos = start_pos.astype(np.float32)
        self.gate_pos = gates_pos.astype(np.float32)
        self.gate_yaw = gate_yaw.astype(np.float32)
        self.num_gates = gates_pos.shape[0]
        self.gates_ahead = gates_ahead
        self.f_func = f_func
        
        # Pause if collision
        self.pause_if_collision = pause_if_collision

        # Calculate relative gates
        # pos,yaw of gate i in reference frame of gate i-1 (assumes a looped track)
        self.gate_pos_rel = np.zeros((self.num_gates,3), dtype=np.float32)
        self.gate_yaw_rel = np.zeros(self.num_gates, dtype=np.float32)
        for i in range(0,self.num_gates):
            self.gate_pos_rel[i] = self.gate_pos[i] - self.gate_pos[i-1]
            # Rotation matrix
            R = np.array([
                [np.cos(self.gate_yaw[i-1]), np.sin(self.gate_yaw[i-1])],
                [-np.sin(self.gate_yaw[i-1]), np.cos(self.gate_yaw[i-1])]
            ])
            self.gate_pos_rel[i,0:2] = R@self.gate_pos_rel[i,0:2]
            self.gate_yaw_rel[i] = self.gate_yaw[i] - self.gate_yaw[i-1]

        # Define the target gate for each environment
        self.target_gates = np.zeros(num_envs, dtype=int)

        # action space: [p_cmd,q_cmd,r_cmd,T_cmd] normalized to [-1,1]
        action_space = spaces.Box(low=-1, high=1, shape=(4,))

        # observation space: pos[G], vel[G], att[eulerB->G], rates[B], rpms, future_gates[G], future_gate_dirs[G]
        # [G] = reference frame aligned with target gate
        # [B] = body frame
        self.state_len = 13+4*self.gates_ahead
        observation_space = spaces.Box(
            low  = np.array([-np.inf]*self.state_len),
            high = np.array([ np.inf]*self.state_len)
        )

        # Initialize the VecEnv
        VecEnv.__init__(self,num_envs, observation_space, action_space)

        # world state: pos[W], vel[W], att[eulerB->W], rates[B], rpms
        self.world_states = np.zeros((num_envs,13), dtype=np.float32)
        # observation state
        self.states = np.zeros((num_envs,self.state_len), dtype=np.float32)

        # Define any other environment-specific parameters
        self.max_steps = 1200      # Maximum number of steps in an episode
        self.dt = np.float32(0.01) # Time step duration

        self.step_counts = np.zeros(num_envs, dtype=int)
        self.actions = np.zeros((num_envs,4), dtype=np.float32)
        self.dones = np.zeros(num_envs, dtype=bool)
        self.final_gate_passed = np.zeros(num_envs, dtype=bool)

        self.update_states = self.update_states_gate

        self.pause = False

    def update_states_world(self):
        self.states = self.world_states

    def update_states_gate(self):
        # Transform pos and vel in gate frame
        gate_pos = self.gate_pos[self.target_gates%self.num_gates]
        gate_yaw = self.gate_yaw[self.target_gates%self.num_gates]

        # Rotation matrix from world frame to gate frame
        R = np.array([
            [np.cos(gate_yaw), np.sin(gate_yaw)],
            [-np.sin(gate_yaw), np.cos(gate_yaw)]
        ]).transpose((2,1,0))

        # new state array to prevent the weird bug related to indexing ([:] syntax)
        new_states = np.zeros_like(self.states)

        # Update positions
        pos_W = self.world_states[:,0:3]
        pos_G = (pos_W[:,np.newaxis,0:2] - gate_pos[:,np.newaxis,0:2]) @ R
        new_states[:,0:2] = pos_G[:,0,:]
        new_states[:,2] = pos_W[:,2] - gate_pos[:,2]

        # Update velocities
        vel_W = self.world_states[:,3:6]
        vel_G = (vel_W[:,np.newaxis,0:2]) @ R
        new_states[:,3:5] = vel_G[:,0,:]
        new_states[:,5] = vel_W[:,2]

        # Update attitude
        new_states[:,6:8] = self.world_states[:,6:8]
        yaw = self.world_states[:,8] - gate_yaw
        yaw %= 2*np.pi
        yaw[yaw > np.pi] -= 2*np.pi
        yaw[yaw < -np.pi] += 2*np.pi
        new_states[:,8] = yaw

        # Update rates
        new_states[:,9:12] = self.world_states[:,9:12]

        # Update Thrust
        new_states[:,12] = self.world_states[:,12]

        # Update future gates relative to current gate ([0,0,0,0] for out of bounds)
        for i in range(self.gates_ahead):
            indices = (self.target_gates+i+1)
            # loop when out of bounds
            indices = indices % self.num_gates
            valid = indices < self.num_gates
            new_states[valid,13+4*i:13+4*i+3] = self.gate_pos_rel[indices[valid]]
            new_states[valid,13+4*i+3] = self.gate_yaw_rel[indices[valid]]

        self.states = new_states

    def reset_(self, dones):
        num_reset = dones.sum()

        x0 = np.random.uniform(-0.5,0.5, size=(num_reset,)) + self.start_pos[0]
        y0 = np.random.uniform(-0.5,0.5, size=(num_reset,)) + self.start_pos[1]
        z0 = np.random.uniform(-0.5,0.5, size=(num_reset,)) + self.start_pos[2]
        
        vx0 = np.random.uniform(-0.5,0.5, size=(num_reset,))
        vy0 = np.random.uniform(-0.5,0.5, size=(num_reset,))
        vz0 = np.random.uniform(-0.5,0.5, size=(num_reset,))
        
        phi0   = np.random.uniform(-np.pi/9,np.pi/9, size=(num_reset,))
        theta0 = np.random.uniform(-np.pi/9,np.pi/9, size=(num_reset,))
        psi0   = np.random.uniform(-np.pi,np.pi, size=(num_reset,))
        
        p0 = 0.1*np.random.randn(num_reset)
        q0 = 0.1*np.random.randn(num_reset)
        r0 = 0.1*np.random.randn(num_reset)
        T0 = np.random.uniform(-.1,.1, size=(num_reset,))

        self.world_states[dones] = np.stack([x0, y0, z0, vx0, vy0, vz0, phi0, theta0, psi0, p0, q0, r0, T0], axis=1)

        self.step_counts[dones] = np.zeros(num_reset)
        
        self.target_gates[dones] = np.zeros(num_reset, dtype=int)

        # update states
        self.update_states()
        return self.states
    
    def reset(self):
        return self.reset_(np.ones(self.num_envs, dtype=bool))

    def step_async(self, actions):
        self.actions = actions
    
    def step_wait(self):
        new_states = self.world_states + self.dt*self.f_func(self.world_states.T, self.actions.T).T
        self.step_counts += 1

        pos_old = self.world_states[:,0:3]
        pos_new = new_states[:,0:3]
        pos_gate = self.gate_pos[self.target_gates%self.num_gates]
        yaw_gate = self.gate_yaw[self.target_gates%self.num_gates]

        # Rewards from [...]
        d2g_old = np.linalg.norm(pos_old - pos_gate, axis=1)
        d2g_new = np.linalg.norm(pos_new - pos_gate, axis=1)
        rat_penalty = 0.01*np.linalg.norm(new_states[:,9:12], axis=1)
        #Yaw Penalty
        yaw_penalty = (new_states[:, 8]-yaw_gate)**2

        rewards = d2g_old - d2g_new - rat_penalty - yaw_penalty

        
        normal = np.array([np.cos(yaw_gate), np.sin(yaw_gate)]).T
        # dot product of normal and position vector over axis 1
        pos_old_projected = (pos_old[:,0]-pos_gate[:,0])*normal[:,0] + (pos_old[:,1]-pos_gate[:,1])*normal[:,1]
        pos_new_projected = (pos_new[:,0]-pos_gate[:,0])*normal[:,0] + (pos_new[:,1]-pos_gate[:,1])*normal[:,1]
        passed_gate_plane = (pos_old_projected < 0) & (pos_new_projected > 0)
        gate_passed = passed_gate_plane & np.all(np.abs(pos_new - pos_gate)<0.5, axis=1)
        gate_collision = passed_gate_plane & np.any(np.abs(pos_new - pos_gate)>0.5, axis=1)
        
        # gate reward (+ dist penalty)
        rewards[gate_passed] = 10 #- 10*d2g_new[gate_passed]
        
        # Check for gate collision
        rewards[gate_collision] = -10

        # Check ground collision (z > 0)
        ground_collision = new_states[:,2] > 0
        rewards[ground_collision] = -10
        
        # Check out of bounds
        # outside grid abs(x,y)>10
        # prevent numerical issues: abs(p,q,r) < 1000
        out_of_bounds = np.any(np.abs(new_states[:,0:2]) > 10, axis=1) | np.any(np.abs(new_states[:,9:12]) > 1000, axis=1)
        rewards[out_of_bounds] = -10
        
        # Check number of steps
        max_steps_reached = self.step_counts >= self.max_steps

        # Update target gate
        self.target_gates[gate_passed] += 1
        self.target_gates[gate_passed] %= self.num_gates
        
        # Check if final gate has been passed
        # self.final_gate_passed = self.target_gates >= self.num_gates

        # give reward for passing final gate
        rewards[self.final_gate_passed] = 10
        
        # Check if the episode is done
        dones = max_steps_reached | ground_collision | gate_collision | out_of_bounds #| self.final_gate_passed
        self.dones = dones
        
        # Pause if collision
        if self.pause:
            dones = dones & ~dones
            self.dones = dones
        elif self.pause_if_collision:
            # dones = max_steps_reached | final_gate_passed | out_of_bounds
            update = ~dones #~(gate_collision | ground_collision)
            # Update world states
            self.world_states[update] = new_states[update]
            self.update_states()
            # Reset env if done (and update states)
            # self.reset_(dones)
        else:
            # Update world states
            self.world_states = new_states
            # reset env if done (and update states)
            self.reset_(dones)


        # Write info dicts
        infos = [{}] * self.num_envs
        for i in range(self.num_envs):
            if dones[i]:
                infos[i]["terminal_observation"] = self.states[i]
            if max_steps_reached[i]:
                infos[i]["TimeLimit.truncated"] = True
        return self.states, rewards, dones, infos
    
    def close(self):
        pass

    def seed(self, seed=None):
        pass

    def get_attr(self, attr_name, indices=None):
        pass

    def set_attr(self, attr_name, value, indices=None):
        pass

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        pass

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False]*self.num_envs

    def render(self, mode=None):
        # Outputs a dict containing all information for rendering
        state_dict = dict(zip(['x','y','z','vx','vy','vz','phi','theta','psi','p','q','r','T'], self.world_states.T))
        # Rescale actions to [0,1] for rendering
        action_dict = dict(zip(['u1','u2','u3','u4'], (np.array(self.actions.T)+1)/2))
        return {**state_dict, **action_dict}