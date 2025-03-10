
import mujoco
import numpy as np
from .base_env import BaseEnv
from gym import spaces
from gym.utils import seeding
import cv2


class BottleCapBaseEnv(BaseEnv):
    def __init__(self, xml_path: str, frame_skip:int, obs_dim, img_w, img_h, visualize:bool):
        self.visualize = visualize
        self.img_w, self.img_h = img_w, img_h
        # self.obs_dim = np.sum(obs_dim)
        # obs = np.random.random(self.obs_dim)
        super(BottleCapBaseEnv, self).__init__(xml_path = xml_path,
                                                frame_skip=frame_skip,
                                                obs_dim=obs_dim)
        self.env_name = xml_path.split('/')[:-4]
        self.act_mid = np.mean(self.model.actuator_ctrlrange, axis=1)
        self.act_rng = 0.5 * (self.model.actuator_ctrlrange[:, 1] - self.model.actuator_ctrlrange[:, 0])
        self.action_space.high = np.ones_like(self.model.actuator_ctrlrange[:, 1])
        self.action_space.low = -1.0 * np.ones_like(self.model.actuator_ctrlrange[:, 0])
        self.info = dict(goal_achieved=False)

    def step(self, a):
        a = np.clip(a, -1.0, 1.0)
        a = self.act_mid + a * self.act_rng  # mean center and scale

        self.do_simulation(a, self.frame_skip)
        obs = self.observations()
        reward, r1 = self.compute_reward()


        if self.visualize:
            # self.render()
            visual_img = self.get_camera_data('ego_camera', self.img_h, self.img_w, 'rgb')
            self.info['imgs'] = visual_img
        self.info['goal_achieved'] = True if r1 > np.pi else False
        done = True if r1 > np.pi else False
        # done = False
        # if done:
        #     reward += 10
        return obs, reward, done, done, self.info
    def reset(self, seed=None, options = None):
        # ob, _ = super(BottleCapEnv, self).reset(seed=seed)
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)

        self._reset_simulation()

        ob = self.reset_model()
        # if self.render_mode == "human" and self.visualize:
        #     self.render()
        return ob, {}

    def reset_model(self):

        qp = self.init_qpos.copy()
        qv = self.init_qvel.copy()
        qp[:-1] += self.np_random.normal(0, 0.002, self.init_qpos.shape[0]-1)
        qv[:-1] += self.np_random.normal(0, 0.001, self.init_qpos.shape[0]-1)
        self.set_state(qp, qv)

        return self.observations()

    def observations(self, obs=None):
        joint_states = self.state_vector().copy()
        obs = joint_states.copy()
        assert obs.shape[0] == self.obs_dim, f'len of obs is {obs.shape[0]}, but defined {self.obs_dim}'
        return obs

    def compute_reward(self):

        objRZ_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'OBJRz')
        r1 = min(self.data.qpos[objRZ_id], 4)
        r2 = self.data.qvel[objRZ_id]

        r = 0.5*r1 + r2

        if r > np.pi:
            r += 5
        if r > 2*np.pi:
            r += 10

        return r, r1

class BottleCapTactileEnv(BottleCapBaseEnv):
    def __init__(self, xml_path: str, frame_skip:int, obs_dim, img_w, img_h, visualize:bool):
        super(BottleCapTactileEnv, self).__init__(xml_path, frame_skip, obs_dim, img_w, img_h, visualize)

    def observations(self, obs=None):
        joint_states = self.state_vector().copy()
        touch_force = np.array(self.data.sensordata.copy()>0.01, dtype=np.float64)  #binary

        obs = np.concatenate([joint_states, touch_force])
        # obs = joint_states.copy()
        assert obs.shape[0] == self.obs_dim, f'len of obs is {obs.shape[0]}, but defined {self.obs_dim}'
        return obs

class BottleCapVisionEnv(BottleCapBaseEnv):
    def __init__(self, xml_path: str, frame_skip:int, obs_dim, img_w, img_h, visualize:bool):
        super(BottleCapVisionEnv, self).__init__(xml_path, frame_skip, obs_dim, img_w, img_h,  visualize)

        self.sp = 0
    def observations(self, obs=None):
        joint_states = self.state_vector().copy()
        image = self.get_camera_data('ego_camera', self.img_w, self.img_h, 'rgb')
        # image = cv2.resize(image, (224, 224))
        # cv2.imwrite(f'./a{self.sp}.png', image)
        # self.sp+=1
        obs = np.concatenate([joint_states, image.flatten()])
        # obs = joint_states.copy()
        assert obs.shape[0] == self.obs_dim, f'len of obs is {obs.shape[0]}, but defined {self.obs_dim}'
        return obs

class BottleCapVTacEnv(BottleCapBaseEnv):
    def __init__(self, xml_path: str, frame_skip:int, obs_dim, img_w, img_h,visualize:bool):
        super(BottleCapVTacEnv, self).__init__(xml_path, frame_skip, obs_dim, img_w, img_h,visualize)
        self.sp=0
    def observations(self, obs=None):
        joint_states = self.state_vector().copy()
        image = self.get_camera_data('ego_camera', self.img_w, self.img_h, 'rgb')
        if self.visualize:
            image = cv2.resize(image, (224, 224))
        # cv2.imwrite(f'./a{self.sp}.png', image)
        # self.sp+=1
        # touch_force = self.data.sensordata.copy() / self.model.sensor_cutoff
        touch_force = np.array(self.data.sensordata.copy() > 0.01, dtype=np.float64) #binary

        obs = np.concatenate([joint_states, image.flatten(), touch_force])
        # obs = joint_states.copy()
        assert obs.shape[0] == self.obs_dim, f'len of obs is {obs.shape[0]}, but defined {self.obs_dim}'
        return obs