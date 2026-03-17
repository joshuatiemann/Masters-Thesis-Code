from gymnasium.envs.mujoco.ant_v4 import AntEnv
import numpy as np
from gymnasium import utils
import os
from scipy import ndimage
from scipy.signal import convolve2d
from gymnasium import spaces
import gymnasium as gym
import mujoco

#Adapted from https://github.com/malteschilling/ddrl

DEFAULT_CAMERA_CONFIG = {
    "distance": 150.0,
    "type": 1,
    "trackbodyid": 0,
    "elevation": 400.0,
}

class QuantrupedEnv(AntEnv):
    def __init__(
        self,
        xml_file="ant.xml",
        energy_saving = True,
        ctrl_cost_weight=0.5, 
        use_contact_forces=False,
        contact_cost_weight=5e-4,
        healthy_reward=0.0,
        render_mode = None,
        terminate_when_unhealthy=True,
        healthy_z_range=(0.2 , 1), 
        contact_force_range=(-1.0, 1.0),
        reset_noise_scale=0.1,
        exclude_current_positions_from_observation=True,
        hf_smoothness=0.25,
        tar_vel = False,
        target_velocity = 1,
        **kwargs
    ):
        super().__init__(
            xml_file=os.path.join(os.path.dirname(__file__), 'assets', 'ant.xml'),
            ctrl_cost_weight=ctrl_cost_weight,
            use_contact_forces=use_contact_forces,
            contact_cost_weight=contact_cost_weight,
            healthy_reward=healthy_reward,
            render_mode = render_mode,
            **kwargs
        )
        self.energy_saving = energy_saving
        self.tar_vel = tar_vel
        self.target_velocity = target_velocity

        self._healthy_z_range = healthy_z_range

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(27,), dtype=np.float64
        )        
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(8,), dtype=np.float64
        )

    def step(self, action):
        xy_position_before = self.get_body_com("torso")[:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.get_body_com("torso")[:2].copy()

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        if self.tar_vel:
            forward_reward =  - abs(x_velocity - self.target_velocity) + self.target_velocity
        else:
            forward_reward = x_velocity
        healthy_reward = self.healthy_reward

        rewards = forward_reward + healthy_reward
        if self.energy_saving:
            costs = ctrl_cost = self.control_cost(action)
        else:
            costs = ctrl_cost = 0
        terminated = self.terminated
        observation = self._get_obs()
        info = {
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_survive": healthy_reward,
            "x_position": xy_position_after[0],
            "y_position": xy_position_after[1],
            "distance_from_origin": np.linalg.norm(xy_position_after, ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            "forward_reward": forward_reward,
        }
        if self._use_contact_forces:
            contact_cost = self.contact_cost
            costs += contact_cost
            info["reward_ctrl"] = -contact_cost

        reward = rewards - costs

        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, False, info

    @property
    def is_healthy(self):
        state = self.state_vector()
        min_z, max_z = self._healthy_z_range
        is_healthy = np.isfinite(state).all() and min_z <= state[2] <= max_z
        return is_healthy
    
    @property
    def healthy_reward(self):
        return (
            float(self.is_healthy or self._terminate_when_unhealthy)
            * self._healthy_reward
        )
    

class Ant(QuantrupedEnv):
    def __init__(self, use_contact_forces=True, *args, **kwargs):
        super().__init__(use_contact_forces=True, *args, **kwargs)
        self.observation_space = spaces.Box(
        low=-np.inf, high=np.inf, shape=(111,), dtype=np.float64
            )        
