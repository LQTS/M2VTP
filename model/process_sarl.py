from model.ppo import PPO
# from stable_baselines3 import PPO


def process_sarl(args, env, cfg_train, logdir):
    learn_cfg = cfg_train["learn"]
    is_testing = learn_cfg["test"]

    if learn_cfg['resume_model'] != '':
        args.resume_model = learn_cfg['resume_model']
    # is_testing = True
    # Override resume and testing flags if they are passed as parameters.
    if args.resume_model != "":
        # is_testing = True
        chkpt_path = args.resume_model

    # if args.max_iterations != -1:
    #     cfg_train["learn"]["max_iterations"] = args.max_iterations

    # logdir = logdir + "/seed{}".format(args.task_envs["seed"])

    """Set up the algo system for training or inferencing."""
    # if args.algo.upper()=='PPO1':
    model = eval(args.algo.upper())(vec_env=env,
                                    cfg_train = cfg_train,
                                    cfg_env = args.task_envs,
                                    sampler=learn_cfg.get("sampler", 'sequential'),
                                    log_dir=logdir,
                                    is_testing=is_testing,
                                    print_log=learn_cfg["print_log"],
                                    device=args.device
                                    )
    # elif args.algo.upper()=='PPO':
    #     model = eval(args.algo.upper())(
    #         actor_critic,
    #         args.clip_param,
    #         args.ppo_epoch,
    #         args.num_mini_batch,
    #         args.value_loss_coef,
    #         args.entropy_coef,
    #         lr=args.lr,
    #         eps=args.eps,
    #         max_grad_norm=args.max_grad_norm)

    # ppo.test("/home/hp-3070/logs/demo/scissors/ppo_seed0/model_6000.pt")
    if is_testing and args.resume_model != "":
        print("Loading model from {}".format(chkpt_path))
        model.test(chkpt_path)
    elif args.resume_model != "":
        print("Loading model from {}".format(chkpt_path))
        model.load(chkpt_path)

    return model

# from bidexhands.algorithms.rl.ppo import PPO
# from bidexhands.algorithms.rl.sac import SAC
# from bidexhands.algorithms.rl.td3 import TD3
# from bidexhands.algorithms.rl.ddpg import DDPG
# from bidexhands.algorithms.rl.trpo import TRPO
# def process_sarl(args, env, cfg_train, logdir):
#     learn_cfg = cfg_train["learn"]
#     is_testing = learn_cfg["test"]
#     # is_testing = True
#     # Override resume and testing flags if they are passed as parameters.
#     if args.model_dir != "":
#         is_testing = True
#         chkpt_path = args.model_dir
#
#     if args.max_iterations != -1:
#         cfg_train["learn"]["max_iterations"] = args.max_iterations
#
#     logdir = logdir + "_seed{}".format(env.task.cfg["seed"])
#
#     """Set up the algo system for training or inferencing."""
#     model = eval(args.algo.upper())(vec_env=env,
#               cfg_train = cfg_train,
#               device=env.rl_device,
#               sampler=learn_cfg.get("sampler", 'sequential'),
#               log_dir=logdir,
#               is_testing=is_testing,
#               print_log=learn_cfg["print_log"],
#               apply_reset=False,
#               asymmetric=(env.num_states > 0)
#               )
#
#     # ppo.test("/home/hp-3070/logs/demo/scissors/ppo_seed0/model_6000.pt")
#     if is_testing and args.model_dir != "":
#         print("Loading model from {}".format(chkpt_path))
#         model.test(chkpt_path)
#     elif args.model_dir != "":
#         print("Loading model from {}".format(chkpt_path))
#         model.load(chkpt_path)
#
#     return model