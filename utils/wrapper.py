
import copy
import os.path

import cv2
import numpy as np
from gymnasium import Wrapper
from stable_baselines3.common.vec_env import  VecEnvWrapper
import torch

class MuJoCoBaseWrapper(Wrapper):
    def __init__(self, env, args):
        super(MuJoCoBaseWrapper, self).__init__(env)

        self.env = env
        self.env_name = env.spec.name
        self.observation_space = copy.deepcopy(self.env.observation_space)
        self.num_envs = 1
        self.args = args

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        # clip obs
        observation = np.clip(observation, -self.args['clip_observations'], self.args['clip_observations'])

        return observation, reward, terminated, truncated, info

    def reset(self, seed = None, options= None):
        reset_obs, _ = self.env.reset(**dict(seed= seed,options= options))

        return reset_obs, _

# refer to https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/envs.py#L167
class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        self.device = device
        # TODO: Fix data types

    def reset(self):
        obs = self.venv.reset()
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs

    def step_async(self, actions):
        if isinstance(actions, torch.LongTensor):
            # Squeeze the dimension for discrete actions
            actions = actions.squeeze(1)
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).float().to(self.device)
        done = torch.from_numpy(done).float().to(self.device)
        info_new = dict(goal_achieved=[], imgs=[])
        for _ in info:
            info_new['goal_achieved'].append(_['goal_achieved'])
            if 'imgs' in _.keys():
                info_new['imgs'].append(_['imgs'])
        info_new['goal_achieved'] = torch.tensor(info_new['goal_achieved']).to(self.device)
        return obs, reward, done, info_new

class VideoWrapper(VecEnvWrapper):
    def __init__(self, venvs, n_frames, video_path):
        super(VideoWrapper, self).__init__(venvs)
        self.venvs = venvs
        self.n_frames = n_frames
        self.env_names = self.venvs.get_attr('env_name')
        # self.recoder = dict(n_frames=0)
        self.recoder = dict()
        self.clear()
        self.video_path = video_path
        os.makedirs(self.video_path, exist_ok=True)
        print(f"start video recoder!!!  Save path is {self.video_path}")

    def reset(self):

        self.clear()
        return self.venvs.reset()

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()

        self.add_frame(info['imgs'])

        # if self.recoder['n_frames'] == self.n_frames or all(done):
        if any(done):

            dones_idx = (done > 0).nonzero(as_tuple=False)[:, 0]

            for id in dones_idx:
                save_name = self.env_names[id]
                # if k == 'n_frames':
                #     continue
                # self.save_video(k, v)
                self.save_video(save_name, self.recoder[save_name])
                self.clear(save_name)

        return obs, reward, done, info

    def clear(self, saved_name=None):
        if saved_name==None:
            for name in self.env_names:
                self.recoder[name] = list()
        else:
            self.recoder[saved_name] = list()

    def add_frame(self, frames):

        for name, f in zip(self.env_names, frames):
            self.recoder[name].append(f)
        # self.recoder['n_frames'] += 1

    def save_video(self, name, imgs):

        size = np.array(imgs[0].shape)[:-1]

        cnt=0
        save_path = os.path.join(self.video_path, name+f'_00{cnt}.mp4')
        while os.path.exists(save_path):
            cnt += 1
            save_path = os.path.join(self.video_path, name + f'_00{cnt}.mp4')

        # set video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4
        videowrite = cv2.VideoWriter(save_path, fourcc, 24, size)

        for i in range(len(imgs)):
            imgs[i] = cv2.resize(imgs[i], size)
            videowrite.write(imgs[i])
        print(f'video {name} has saved in {save_path}')
