import os.path

import gym
import torch
from gym import register
from .wrapper import MuJoCoBaseWrapper, VecPyTorch, VideoWrapper
from gym.utils import seeding
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv


def make_env(env_name, idx, config):
    """return wrapped gym environment for parallel sample collection (vectorized environments)"""
    def helper():
        e = gym.make('{}-v0'.format(env_name))
        e.np_random, seed = seeding.np_random(config['seed']+idx*123)
        return MuJoCoBaseWrapper(e, config)

    return helper

def RegisterEnv(env_name, env_py_path, kwargs):
    register(id=env_name + '-v0',
             entry_point='%s:%s' % (env_py_path, env_name),
             max_episode_steps=200,
             kwargs=kwargs)
def RegisterEnvs(config, test=False):
    root_path = config['env_path']
    env_names = config['env_names'] if not test else config['env_names']+config['eval_env_names']
    env_py = config['env_py_path']
    ep_length = config['ep_length']
    frame_skip = config['frame_skip']
    obs_dim = config['obs_dim']
    visualize = config['visualize']
    img_w = config['img_w']
    img_h = config['img_h']
    for env_name in env_names:
        config_dict = dict(xml_path=os.path.join(root_path, env_name+'.xml'),
                           frame_skip=frame_skip,
                           obs_dim=obs_dim,
                           img_w=img_w,
                           img_h=img_h,
                           visualize=visualize
                           )
        register(id=env_name + '-v0',
                 entry_point=env_py,
                 max_episode_steps=ep_length,
                 kwargs=config_dict)
    return f'successfully register {len(env_names)} envs!!'

# refer to https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/envs.py#L167
def make_vec_envs(env_name,
                  args,
                  num_processes,
                  device,
                  ):
    envs = [make_env(env_name, i, args) for i in range(num_processes)]

    if len(envs) > 1:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)

    envs.seed(args['seed'])
    envs = VecPyTorch(envs, device)
    # if args['visualize']:
    #     envs = VideoWrapper(envs, n_frames=args['ep_length'], video_path=args['video_path'])

    return envs
def make_multi_vec_envs(env_names,
                          args,
                          num_processes,
                          device,
                          task_name
                          ):
    envs = []
    count = 0
    for env_name in env_names:
        for i in range(num_processes):
            envs.append(make_env(env_name, count, args))
            count += 1


    if len(envs) > 1:
         envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)

    envs = VecPyTorch(envs, device)
    envs.seed(args['seed'])

    if args['visualize']:
        envs = VideoWrapper(envs, n_frames=args['ep_length'], video_path=os.path.join(args['video_path'], task_name, str(args['seed'])))

    return envs


