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
    "trackbodyid": 1,
    "elevation": 300.0,
}

def create_new_hfield(mj_model, smoothness = 0.15, bump_scale=2.):
    # Generation of the shape of the height field is taken from the dm_control suite,
    # see dm_control/suite/quadruped.py in the escape task (but we don't use the bowl shape).
    # Their parameters are TERRAIN_SMOOTHNESS = 0.15  # 0.0: maximally bumpy; 1.0: completely smooth.
    # and TERRAIN_BUMP_SCALE = 2  # Spatial scale of terrain bumps (in meters). 
    res = mj_model.hfield_ncol[0]
    row_grid, col_grid = np.ogrid[-1:1:res*1j, -1:1:res*1j]
    # Random smooth bumps.
    terrain_size = 2 * mj_model.hfield_size[0, 0]
    bump_res = int(terrain_size / bump_scale)
    bumps = np.random.uniform(smoothness, 1, (bump_res, bump_res))
    smooth_bumps = ndimage.zoom(bumps, res / float(bump_res))
    # Terrain is elementwise product.
    hfield = (smooth_bumps - np.min(smooth_bumps))[0:mj_model.hfield_nrow[0],0:mj_model.hfield_ncol[0]]
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

class Centipede(AntEnv):
    def __init__(
        self,
        xml_file="Centipede_3.xml",
        energy_saving = True,
        ctrl_cost_weight=0.1, #was 0.5
        use_contact_forces=False,
        contact_cost_weight=5e-4,
        healthy_reward=1,
        render_mode = None,
        terminate_when_unhealthy=True,
        healthy_z_range=(0.251 , 1.25), #body is 0.25 large
        contact_force_range=(-1.0, 1.0),
        reset_noise_scale=0.1,
        exclude_current_positions_from_observation=True,
        hf_smoothness=0.25,
        tar_vel = False,
        target_velocity = 0.75,
        **kwargs
    ):
        if not os.path.isabs(xml_file):
            xml_file = os.path.join(os.path.dirname(__file__), 'assets', xml_file)

        super().__init__(
            xml_file=xml_file,
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

        self._healthy_z_range = healthy_z_range

        obs_shape = self._get_obs().shape
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float64
        )        
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float64
        )


        #ant_mass = mujoco.mj_getTotalmass(self.model)

    def _get_obs(self):
        # Calculate number of segments if not already done
        if not hasattr(self, 'num_segments'):
            self.num_segments = 0
            while mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"torso_{self.num_segments+1}") != -1:
                self.num_segments += 1

        obs_list = []

        # --- 1. Root ---
        # root coords x, y, z (+ quaternion) -> self.data.joint('root').qpos
        # Note: qpos for a free joint is 7 dims (3 pos + 4 quat)
        obs_list.append(self.data.joint("root").qpos)
        # root velocities -> self.data.joint('root').qvel (6 dims: 3 lin + 3 ang)
        obs_list.append(self.data.joint("root").qvel)

        # Legs 0 (Attached to Root)
        for side in ["left", "right"]:
            for joint in ["hip", "ankle"]:
                jname = f"{joint}_{side}_0"
                obs_list.append(self.data.joint(jname).qpos)
                obs_list.append(self.data.joint(jname).qvel)

        # --- 2. Subsequent Segments ---
        for i in range(1, self.num_segments + 1):
            # Torso i
            # Use body ID to fetch global state
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"torso_{i}")
            
            # coords x,y,z (matches root qpos[:3])
            obs_list.append(self.data.xpos[bid])
            # quaternion orientation (matches root qpos[3:])
            obs_list.append(self.data.xquat[bid])
            
            # Velocity (matches root qvel structure: linear, then angular)
            # mujoco cvel is [angular, linear], so we swap parts
            cvel = self.data.cvel[bid]
            obs_list.append(cvel[3:6]) # Linear
            obs_list.append(cvel[0:3]) # Angular

            # Connectors
            for c_axis in ["rl", "ud"]:
                cname = f"connector_{c_axis}_{i}"
                obs_list.append(self.data.joint(cname).qpos)
                obs_list.append(self.data.joint(cname).qvel)

            # Legs i
            for side in ["left", "right"]:
                for joint in ["hip", "ankle"]:
                    jname = f"{joint}_{side}_{i}"
                    obs_list.append(self.data.joint(jname).qpos)
                    obs_list.append(self.data.joint(jname).qvel)
        obs = np.concatenate(obs_list)
        return obs

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
        state = self._get_obs()
        min_z, max_z = self._healthy_z_range
        is_finite = np.isfinite(state).all()

        # Check root torso height
        all_in_range = min_z <= self.data.body("torso").xpos[2] <= max_z

        # Check subsequent torsos
        if hasattr(self, 'num_segments'):
            for i in range(1, self.num_segments + 1):
                z_pos = self.data.body(f"torso_{i}").xpos[2]
                all_in_range = all_in_range and (min_z <= z_pos <= max_z)

        return is_finite and all_in_range
    
    def update_hfield(self, smoothness=1, bump_scale=2):
        if smoothness is not None:
            self.smoothness = smoothness
        if bump_scale is not None:
            self.bump_scale = bump_scale
        create_new_hfield(self.model, smoothness, bump_scale)
        #print("Hfield updated with smoothness: ", self.smoothness, " and bump scale: ", self.bump_scale)  
