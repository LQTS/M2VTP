
import mujoco
from gym import utils, spaces
from gym.envs.mujoco import mujoco_env
from collections import OrderedDict
import numpy as np
from typing import Optional

class BaseEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, xml_path: str, frame_skip: int, obs_dim):
        '''
        :param xml_path: model path
        :param frame_skip: render time step
        :param obs_dim: obs numpy
        :param kwargs:
            render_mode: Optional[str] = None,
            width: int = DEFAULT_SIZE,
            height: int = DEFAULT_SIZE,
            camera_id: Optional[int] = None,
            camera_name: Optional[str] = None,
        '''

        self.metadata = {
            'render_modes': ['human', 'rgb_array', 'depth_array'],
            'render_fps': int(np.round(1.0 / (0.002*frame_skip)))
        }
        # set up obs and space
        random_obs = self.convert_obs_dim_to_obs(obs_dim=obs_dim)
        self.obs_dim = random_obs.shape[0]
        observation_space = self.convert_observation_to_space(random_obs)

        mujoco_env.MujocoEnv.__init__(self, model_path = xml_path,
                                      frame_skip = frame_skip,
                                      observation_space = observation_space,
                                      render_mode=self.metadata['render_modes'][0])
        utils.EzPickle.__init__(self)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        raise  NotImplementedError
    def step(self, a):

        raise NotImplementedError

    def reset_model(self):
        raise NotImplementedError

    def observations(self):
        raise NotImplementedError

    def compute_reward(self):
        raise NotImplementedError
    def get_stete(self):
        raise NotImplementedError

    def get_camera_data(self, camera_name: str, width: int, height: int, mode: str):

        self.model.vis.global_.offwidth = width
        self.model.vis.global_.offheight = height

        assert camera_name is not None, 'camera_name is required'
        assert mode in ['rgb', 'depth', 'rgb-d'], 'mode must be in [rgb, depth, rgb-d], current mode is {}'.format(mode)

        camera_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_CAMERA,
            camera_name,
        )
        self._get_viewer('rgb_array').render(camera_id=camera_id)

        if mode == 'rgb':
            data = self._get_viewer('rgb_array').read_pixels(depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == 'depth':
            # Extract depth part of the read_pixels() tuple
            data = self._get_viewer("depth_array").read_pixels(depth=True)[1]
            # original image is upside-down, so flip it
            return data[::-1, :]
        else:
            rgb, dep = self._get_viewer("depth_array").read_pixels(depth=True)

            return rgb[::-1, :, :], dep[::-1, :]

    def visualize_windows(self):
        self.render_mode = 'human'
        self.render()


    def convert_observation_to_space(self, observation):
        if isinstance(observation, dict):
            space = spaces.Dict(OrderedDict([
                (key, self.convert_observation_to_space(value))
                for key, value in observation.items()
            ]))
        elif isinstance(observation, np.ndarray):
            low = np.full(observation.shape, -float('inf'))
            high = np.full(observation.shape, float('inf'))
            space = spaces.Box(low, high, dtype=observation.dtype)
        else:
            raise NotImplementedError(type(observation), observation)

        return space
    # def convert_obs_dim_to_obs(self, obs_dim):
    #
    #     if isinstance(obs_dim, dict):
    #         obs = dict()
    #         for k, v in obs_dim.items():
    #             obs[k] = k = np.random.random(v)
    #     elif isinstance(obs_dim, int):
    #         obs = np.random.random(obs_dim)
    #     else:
    #         raise NotImplementedError(type(obs_dim), obs_dim)
    #     return obs
    def convert_obs_dim_to_obs(self, obs_dim):
        if isinstance(obs_dim, dict):
            all_dim = 0
            for k, v in obs_dim.items():
                all_dim += v
            obs = np.random.random(all_dim)
        elif isinstance(obs_dim, int):
            obs = np.random.random(obs_dim)
        else:
            raise NotImplementedError(type(obs_dim), obs_dim)
        return obs