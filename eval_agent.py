import os

from tqdm import tqdm

from utils.logger import DataLog
os.environ['MUJOCO_GL']='egl' #osmesa  egl   glfw
# os.environ["MUJOCO_EGL_DEVICE_ID"] = "1"
# os.environ['MUJOCO_GL']='glfw'

from config.argument import get_args
from utils.mujoco_utils import RegisterEnvs, make_env, make_vec_envs, make_multi_vec_envs
from model.process_sarl import process_sarl
import torch
import numpy as np

args = get_args()
RegisterEnvs(args.task_envs, test=True)

def parse_task(env_config, device, n_per_env, multi_envs=False, envs_names=None):

    if multi_envs:
        env = make_multi_vec_envs(envs_names, env_config, n_per_env, device, args.task)
    else:
        # env = make_env(env_config['env_names'][0], 0, env_config)
        env = make_vec_envs(env_config['env_names'][0], env_config, n_per_env, device)
    return env

def main(args):

    assert args.test, "please run 'python eval_agent.py --test' in your console."
    args.models["learn"]["n_per_env"] = 4 if not args.task_envs['visualize'] else 1
    # set up envs
    env = parse_task(args.task_envs, device=args.device,n_per_env=args.models["learn"]["n_per_env"], multi_envs=True, envs_names=args.task_envs['env_names'])
    env_eval = parse_task(args.task_envs, device=args.device, n_per_env=args.models["learn"]["n_per_env"], multi_envs=True, envs_names=args.task_envs['eval_env_names'])
    logger = DataLog()

    # set up policy
    sarl = process_sarl(args, env, args.models, args.logger_dir)
    reward, success = sarl.eval(100)
    logger.log_kv('reward', reward)
    logger.log_kv('success_rate', success)
    del sarl

    sarl_eval = process_sarl(args, env_eval, args.models, args.logger_dir)
    reward_eval, success_eval  = sarl_eval.eval(100)
    logger.log_kv('reward_eval', reward_eval)
    logger.log_kv('success_rate_eval', success_eval)
    del sarl_eval

    save_path = os.path.dirname(args.resume_model)
    print(save_path)
    logger.save_log(save_path, 'evaluation')


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