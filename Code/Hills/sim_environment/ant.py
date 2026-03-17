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

def create_new_hfield(mj_model, smoothness = 0.15, bump_scale=2., seed=None):
    # Generation of the shape of the height field is taken from the dm_control suite,
    # see dm_control/suite/quadruped.py in the escape task (but we don't use the bowl shape).
    # Their parameters are TERRAIN_SMOOTHNESS = 0.15  # 0.0: maximally bumpy; 1.0: completely smooth.
    # and TERRAIN_BUMP_SCALE = 2  # Spatial scale of terrain bumps (in meters). 
    res = mj_model.hfield_ncol[0]
    
    # Random smooth bumps.
    terrain_size = 2 * mj_model.hfield_size[0, 0]
    bump_res = int(terrain_size / bump_scale)
    if seed is not None:
        rng = np.random.RandomState(seed)
        bumps = rng.uniform(smoothness, 1, (bump_res, bump_res))
    else:
        bumps = np.random.uniform(smoothness, 1, (bump_res, bump_res))
    smooth_bumps = ndimage.zoom(bumps, res / float(bump_res))
    
    # Terrain is the smooth bumps directly.
    # We remove the normalization (x - min(x)) to preserve the amplitude scaling of 'smoothness'.
    hfield = smooth_bumps[0:mj_model.hfield_nrow[0], 0:mj_model.hfield_ncol[0]]
    
    # Clears a patch shaped like box, assuming robot is placed in center of hfield.
    # Function was implemented in an old rllab version.
    h_center = int(0.5 * hfield.shape[0])
    w_center = int(0.5 * hfield.shape[1])
    patch_size = 2
    fromrow, torow = h_center - int(0.5*patch_size), h_center + int(0.5*patch_size)
    fromcol, tocol = w_center - int(0.5*patch_size), w_center + int(0.5*patch_size)
    # convolve to smoothen edges somewhat, in case hills were cut off
    K = np.ones((patch_size,patch_size)) / patch_size**2
    s = convolve2d(hfield[fromrow-(patch_size-1):torow+(patch_size-1), fromcol-(patch_size-1):tocol+(patch_size-1)], K, mode='same', boundary='symm')
    hfield[fromrow-(patch_size-1):torow+(patch_size-1), fromcol-(patch_size-1):tocol+(patch_size-1)] = s
    # Last, we lower the hfield so that the centre aligns at zero height
    # (importantly, we use a constant offset of -0.5 for rendering purposes)
    #print(np.min(hfield), np.max(hfield))
    hfield = hfield - np.max(hfield[fromrow:torow, fromcol:tocol])
    mj_model.hfield_data[:] = hfield.ravel()

class QuantrupedEnv(AntEnv):
    def __init__(
        self,
        xml_file="ant.xml",
        energy_saving = True,
        ctrl_cost_weight=0.5, #was 0.5
        use_contact_forces=False,
        contact_cost_weight=5e-4,
        healthy_reward=0.0,
        render_mode = None,
        terminate_when_unhealthy=True,
        healthy_z_range=(-2 , 10), #ant is almost unkillable, so it can climb hills without leaving healthy z range
        contact_force_range=(-1.0, 1.0),
        reset_noise_scale=0.1,
        exclude_current_positions_from_observation=True,
        hf_smoothness=0.25,
        tar_vel = False,
        target_velocity = 1,
        hfield_seed = None,
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
        self.target_velocity = target_velocity,
        self.hfield_seed = hfield_seed

        self._healthy_z_range = healthy_z_range

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(27,), dtype=np.float64
        )        
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(8,), dtype=np.float64
        )
        
       # self.update_hfield(0.7, 1)


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
    
    def update_hfield(self, smoothness=1, bump_scale=1):
        if smoothness is not None:
            self.smoothness = smoothness
        if bump_scale is not None:
            self.bump_scale = bump_scale
        create_new_hfield(self.model, smoothness, bump_scale, seed=self.hfield_seed)
        print("Hfield updated with smoothness: ", self.smoothness, " and bump scale: ", self.bump_scale)

        if hasattr(self, "mujoco_renderer") and self.mujoco_renderer is not None:
            for viewer in self.mujoco_renderer._viewers.values():
                if hasattr(viewer, "con"):
                    mujoco.mjr_uploadHField(self.model, viewer.con, 0)
                elif hasattr(viewer, "ctx"):
                    mujoco.mjr_uploadHField(self.model, viewer.ctx, 0)

    def set_hfield(self, hfield_data):
        """Sets the hfield directly from data (copy optimization) and uploads to renderer."""
        self.model.hfield_data[:] = hfield_data
        
        if hasattr(self, "mujoco_renderer") and self.mujoco_renderer is not None:
            for viewer in self.mujoco_renderer._viewers.values():
                if hasattr(viewer, "con"):
                    mujoco.mjr_uploadHField(self.model, viewer.con, 0)
                elif hasattr(viewer, "ctx"):
                    mujoco.mjr_uploadHField(self.model, viewer.ctx, 0)


class Ant(QuantrupedEnv):
    def __init__(self, use_contact_forces=True, *args, **kwargs):
        super().__init__(use_contact_forces=True, *args, **kwargs)
        self.observation_space = spaces.Box(
        low=-np.inf, high=np.inf, shape=(111,), dtype=np.float64
            )        
