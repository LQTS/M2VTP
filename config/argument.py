import os
import argparse
import yaml
from utils.util import pretty
from utils.util import DotDict
import torch
def get_args():
    parser = argparse.ArgumentParser(description='manipulation tasks')

    parser.add_argument('--task', type=str, required=True, help='manipulated task for an agent')
    parser.add_argument('--algo', type=str, default='PPO', help='training algorithm')
    parser.add_argument('--resume_model', type=str, default='', help='Choose a model dir')
    parser.add_argument('--device', type=str, default='cuda:0', help='device for training')
    parser.add_argument('--env_vis', action='store_true', help='visualize envs or not')
    parser.add_argument('--test', action='store_true', help='test')
    parser.add_argument('--seed',required=True,  type=int, help='test')

    args = parser.parse_args()

    assert args.task + '.yaml' in os.listdir('./config/task_env'), f"task is not defined, please choose one from {os.listdir('./config/task_env')}(except '.yaml')"
    with open('./config/task_env/'+args.task+'.yaml') as f:
        args.task_envs = yaml.load(f, Loader=yaml.FullLoader)
    args.task_envs['seed'] = args.seed
    # assert len(args.task_envs['env_names']) == args.task_envs['env_num']
    assert len(args.task_envs['env_names']) == args.task_envs['env_num']
    args.task_envs['env_path'] = os.path.join(os.getcwd(), args.task_envs['env_path'])
    # load model params
    with open(f'./config/algos/ppo/{args.task}.yaml') as f:
        args.models = yaml.load(f, Loader=yaml.FullLoader)
    if args.test:
        args.models['learn']['test'] = args.test
        args.models["learn"]["n_per_env"] = 1
        if args.env_vis:
            args.task_envs['img_w'] = 1080
            args.task_envs['img_h'] = 1080
        # args.task_envs['ep_length'] = 500
    # logger dir
    curr_dir = os.getcwd()
    args.logger_dir = os.path.join(curr_dir, 'runs', args.task, 'seed'+str(args.task_envs['seed']))
    os.makedirs(args.logger_dir, exist_ok=True)

    # args.task_envs, args.models = DotDict(args.task_envs), DotDict(args.models)
    #str->torch.device
    # args.device = torch.device(args.device)
    # visualize envs
    args.task_envs['visualize'] = args.env_vis
    del args.env_vis

    # args.train_params = args.task_envs['train_params']
    # del args.task_envs['train_params']
    if not args.models['learn']['test']:
        config_params = vars(args)
        # save all config
        with open(os.path.join(args.logger_dir, 'all_config.yaml'), 'w') as f:
            f.write(yaml.dump(config_params))

        # print(pretty(config_params))
    return args