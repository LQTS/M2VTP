
import mujoco
import numpy as np
from .base_env import BaseEnv
from gym import spaces

class ShadowHandEnv(BaseEnv):
    def __init__(self, xml_path: str, config):
        self.config = config
        obs = np.random.random(config.obs_dim)
        super(ShadowHandEnv, self).__init__(xml_path = xml_path,
                                            frame_skip=config.frame_skip,
                                            obs_space=obs)

    def step(self, a):

        self.do_simulation(a, self.frame_skip)
        obs = self.observations()
        reward = self.compute_reward()
        done = False

        return obs, reward, done, done, {}

    def reset_model(self):

        qp = self.init_qpos.copy()
        qv = self.init_qvel.copy()
        self.set_state(qp, qv)

        return self.observations()

    def observations(self, obs=None):
        obs = self.state_vector()

        return obs

    def compute_reward(self):

        r = 1+1
        return r

