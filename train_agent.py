import os
os.environ['MUJOCO_GL']='egl'
os.environ["MUJOCO_EGL_DEVICE_ID"] = "1"
# os.environ['MUJOCO_GL']='glfw'
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from config.argument import get_args
from utils.mujoco_utils import RegisterEnvs, make_vec_envs, make_multi_vec_envs
from model.process_sarl import process_sarl
import torch
import numpy as np

args = get_args()
RegisterEnvs(args.task_envs)

def parse_task(env_config, device, n_per_env, multi_envs=False):

    if multi_envs:
        env = make_multi_vec_envs(env_config['env_names'], env_config, n_per_env, device, args.task)
    else:
        # env = make_env(env_config['env_names'][0], 0, env_config)
        env = make_vec_envs(env_config['env_names'][0], env_config, n_per_env, device)
    return env

def main(args):

    # set up envs
    env = parse_task(args.task_envs, device=args.device,n_per_env=args.models["learn"]["n_per_env"], multi_envs=True)
    # set up policy
    sarl = process_sarl(args, env, args.models, args.logger_dir)

    sarl.run(num_learning_iterations=args.models["learn"]["max_iterations"], log_interval=args.models["learn"]["save_interval"])


def set_seed():
    if args.models['seed'] is not None:
        seed = args.models['seed']
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    set_seed()
    main(args)