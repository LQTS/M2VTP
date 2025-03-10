import os.path

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy
import csv
from utils.logger import DataLog
from matplotlib.pyplot import MultipleLocator
import argparse
# import bar_chart_race as bcr
import pandas as pd

def make_train_plots(log = None,
                     log_path = None,
                     keys = None,
                     save_loc = None,
                     sample_key = 'num_samples',
                     x_scale = 1.0,
                     y_scale = 1.0):
    if log is None and log_path is None:
        print("Need to provide either the log or path to a log file")
    if log is None:
        logger = DataLog()
        logger.read_log(log_path)
        log = logger.log
    # make plots for specified keys
    for key in keys:
        if key in log.keys():
            fig = plt.figure(figsize=(10,6))
            ax1 = fig.add_subplot(111)
            try:
                cum_samples = [np.sum(log[sample_key][:i]) * x_scale for i in range(len(log[sample_key]))]
                ax1.plot(cum_samples, [elem * y_scale for elem in log[key]])
                ax1.set_xlabel('samples')
                # mark iteration on the top axis
                ax2 = ax1.twiny() 
                ax2.set_xlabel('iterations', color=(.7,.7,.7))
                ax2.tick_params(axis='x', labelcolor=(.7,.7,.7))
                ax2.set_xlim([0, len(log[key])])
            except:
                ax1.plot(log[key])
                ax1.set_xlabel('iterations')
            ax1.set_title(key)
            plt.savefig(save_loc+'/'+key+'.png', dpi=100)
            plt.close()
def make_mean_std_plot(parent_log_dir_all=None,
                       log_dir_list_all = None,
                       key = None,
                       csv_name=None,
                       save_loc = None,):
    assert parent_log_dir_all is not None
    assert log_dir_list_all is not None
    assert key is not None

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    # fig1 =
    # ax1 = fig.add_subplot(122)
    labels = [task.split('/')[-1][11:].split('+')[0] for task in parent_log_dir_all]
    colors = ['crimson', 'royalblue', 'gold', 'black', 'seagreen', 'darkcyan', 'darkviolet']
    # labels = ['Ours', 'V-Only', 'MVP', 'VT-Sep', 'T-Only']
    labels = ['Ours', 'VT-Scr-R', 'VT-Scr-C']
    if not isinstance(parent_log_dir_all, list):
        parent_log_dir_all = [parent_log_dir_all]
        log_dir_list_all = [log_dir_list_all]
    for i, (parent_log_dir, log_dir_list) in enumerate(zip(parent_log_dir_all, log_dir_list_all)):
        log_path_list = [os.path.join(parent_log_dir, log_dir+'.csv') for log_dir in log_dir_list]
        key_data = []
        for log_path in log_path_list:
            logger = DataLog()
            logger.read_log(log_path)
            log = logger.log
            if key in log.keys():
                key_data.append(log[key])
            else:
                raise KeyError
        min_len = min(min([np.array(data).shape[-1] for data in key_data]), 600)
        curve_data = np.array([np.array(data[:min_len]) for data in key_data])
        curve_data = scipy.signal.savgol_filter(curve_data,60,2)

        mean = np.mean(curve_data, axis=0)
        std = curve_data.std(axis=0) / np.sqrt(curve_data.shape[0])

        x_axis = np.arange(min_len)
        ax.plot(x_axis, mean, label=labels[i], linewidth=3)
        plt.fill_between(x_axis, mean + std, mean - std, alpha=0.2)
        # ax.plot(x_axis, mean, linewidth=2.5)
        # ax.fill_between(x_axis, mean + std, mean - std, alpha=0.3)
    # plt.title('performance of different methods', fontsize=20)
    plt.grid(True, which='both')
    plt.ylabel('Success Rate(%)', fontsize=20)
    # plt.ylabel('Mean Rewards', fontsize=20)
    plt.xlabel('Iteration', fontsize=20)
    plt.xticks(size=14)
    plt.yticks(size=14)
    if 'success' in key:
        y_major_locator = MultipleLocator(10)
        # 把y轴的刻度间隔设置为10，并存在变量里
        ax = plt.gca()
        # ax为两条坐标轴的实例
        # 把x轴的主刻度设置为1的倍数
        ax.yaxis.set_major_locator(y_major_locator)

    plt.legend(prop = {'size':14})
    plt.savefig('../runs/curve/plot/%s0916-EXP1.png' % (csv_name[4:-4]+'_'+key.split('/')[1]), dpi=1000)
    plt.show()
def make_obj_plot(parent_log_dir_all=None,
                       log_dir_list_all = None,
                       data=None,
                  metric=None):
    assert parent_log_dir_all is not None
    assert log_dir_list_all is not None
    assert data is not None

    # colors = ['crimson', 'royalblue', 'gold', 'black', 'seagreen', 'darkcyan', 'darkviolet']

    # labels = ['Ours', 'HOI+P-Glo+P-Loc', 'HOI+P-Glo', 'P-Glo', 'P-Loc', 'HOI', 'Hand2obj']

    if not isinstance(parent_log_dir_all, list):
        parent_log_dir_all = [parent_log_dir_all]
        log_dir_list_all = [log_dir_list_all]
    logger = DataLog()
    logger.read_log('../../grasp_envs/DAPG/exp_all_pointnetL_pre_pct64_0512/seed231bc125/grab/log_evaluation_100_30_grab.csv')
    log = logger.log
    keys = list(log.keys())[2:]

    logger1 = DataLog()
    logger1.read_log('../../grasp_envs/DAPG/exp_all_pointnetL_pre_pct64_0512/seed231bc125/3dnet/log_evaluation_100_30_3dnet.csv')
    log1 = logger1.log
    keys1 = list(log1.keys())[2:]
    for d in data:
        if d == 'grab':
            keys = keys
        elif d == '3dnet':
            keys = keys1
        for key in keys:
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111)
            # fig1 =
            # ax1 = fig.add_subplot(122)
            colors = ['crimson', 'royalblue', 'gold', 'black', 'seagreen']
            labels = [task.split('_')[-2] for task in parent_log_dir_all]
            for i, (parent_log_dir, log_dir_list) in enumerate(zip(parent_log_dir_all, log_dir_list_all)):


                log_path_list = [os.path.join(parent_log_dir, log_dir, d) for log_dir in log_dir_list]
                for l in range(len(log_path_list)):
                    for csv in os.listdir(log_path_list[l]):
                        if metric in csv and '.csv' == csv[-4:]:
                            log_path_list[l] = os.path.join(log_path_list[l], csv)

                key_data = []
                for log_path in log_path_list:
                    logger = DataLog()
                    logger.read_log(log_path)
                    log = logger.log

                    if key in log.keys():
                        key_data.append(log[key])
                    else:
                        raise KeyError
                min_len = min(min([np.array(data).shape[-1] for data in key_data]), 5)
                curve_data = np.array([np.array(data[:min_len]) for data in key_data])


                mean = np.mean(curve_data, axis=0)
                std = curve_data.std(axis=0) / np.sqrt(curve_data.shape[0])

                x_axis = log['it'][:min_len]
                if d == 'grab':
                    ax.plot(x_axis, mean, label=labels[i]+'_'+d, color=colors[i], linestyle='--',marker='o')

                else:
                    ax.plot(x_axis, mean, label=labels[i]+'_'+d, color=colors[i], linestyle='-.',marker='>')
                plt.fill_between(x_axis, mean + std, mean - std, alpha=0.2)
            # ax.plot(x_axis, mean, linewidth=2.5)
            # ax.fill_between(x_axis, mean + std, mean - std, alpha=0.3)
            plt.title(key, fontsize=20)
            plt.grid(True, which='both')
            plt.ylabel('Success Rate(%)', fontsize=20)
            plt.xlabel('Iteration', fontsize=20)
            plt.xticks(size=14)
            plt.yticks(size=14)

            y_major_locator = MultipleLocator(10)
            # 把y轴的刻度间隔设置为10，并存在变量里
            ax = plt.gca()
            # ax为两条坐标轴的实例
            # 把x轴的主刻度设置为1的倍数
            ax.yaxis.set_major_locator(y_major_locator)

            plt.legend()
            savefig_path = '../../grasp_envs/DAPG/plot/'
            os.makedirs(savefig_path, exist_ok=True)
            plt.savefig(savefig_path+'%s_%s_%s.png' % (d, metric, key), dpi=500)
            # plt.show()
def make_mean_std_plot_one_exp(parent_log_dir_all=None,
                       log_dir_list_all = None,
                       key = None,
                       csv_name=None,
                       save_loc = None,):
    assert parent_log_dir_all is not None
    assert log_dir_list_all is not None
    assert key is not None

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    labels = [task[0][6:] for task in log_dir_list_all]
    if not isinstance(parent_log_dir_all, list):
        parent_log_dir_all = [parent_log_dir_all]
        log_dir_list_all = [log_dir_list_all]
    for i, (parent_log_dir, log_dir_list) in enumerate(zip(parent_log_dir_all, log_dir_list_all)):
        log_path_list = [os.path.join(parent_log_dir, log_dir, 'logs', csv_name) for log_dir in log_dir_list]
        key_data = []
        for log_path in log_path_list:
            logger = DataLog()
            logger.read_log(log_path)
            log = logger.log
            if key in log.keys():
                key_data.append(log[key])
            else:
                raise KeyError
        min_len = min(min([np.array(data).shape[-1] for data in key_data]), 800)
        curve_data = np.array([np.array(data[:min_len]) for data in key_data])

        mean = np.mean(curve_data, axis=0)
        std = curve_data.std(axis=0) / np.sqrt(curve_data.shape[0])

        x_axis = np.arange(min_len)
        # ax.plot(x_axis, mean, color=colors[j], label=labels[j])
        # plt.fill_between(x_axis, mean + std, mean - std, color=colors[j], alpha=0.2)
        ax.plot(x_axis, mean, label=labels[i])
        plt.fill_between(x_axis, mean + std, mean - std, alpha=0.2)
    plt.title('%s performance of BC model' % (parent_log_dir_all[0].split('/')[-1][4:]), fontsize=20)
    plt.grid(True, which='both')
    plt.ylabel(key, fontsize=16)
    plt.xlabel('Iteration', fontsize=16)
    if 'success' in key:
        y_major_locator = MultipleLocator(10)
        # 把y轴的刻度间隔设置为10，并存在变量里
        ax = plt.gca()
        # ax为两条坐标轴的实例
        # 把x轴的主刻度设置为1的倍数
        ax.yaxis.set_major_locator(y_major_locator)

    plt.legend()
    plt.savefig('/remote-home/share/lqt/grasp_contactmap14/grasp_envs/DAPG/%s_%s0801.png' % (key, parent_log_dir_all[0].split('/')[-1][4:]), dpi=100)
    plt.show()
def make_bar_plot(parent_log_dir_all=None,
                log_dir_list_all = None,
                key = None,
                csv_name=None,
                it=0):
    assert parent_log_dir_all is not None
    assert log_dir_list_all is not None
    assert key is not None

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)

    x = np.arange(17)
    total_width, n = 0.6, 4
    width = total_width / n
    x = x - (total_width - width) / 2

    labels = [task.split('/')[-1][3:-4] for task in parent_log_dir_all]
    if not isinstance(parent_log_dir_all, list):
        parent_log_dir_all = [parent_log_dir_all]
        log_dir_list_all = [log_dir_list_all]
    for i, (parent_log_dir, log_dir_list) in enumerate(zip(parent_log_dir_all, log_dir_list_all)):
        log_path_list = [os.path.join(parent_log_dir, log_dir, 'logs', csv_name) for log_dir in log_dir_list]
        key_data = []

        success_rate_per = []
        for log_path in log_path_list:
            success = []
            logger = DataLog()
            logger.read_log(log_path)
            log = logger.log
            obj_name = []
            for k, v in log.items():
                if 'success' in k:
                    obj_name.append(k.split('_')[0])
                    success.append(v[it])

            success_rate_per.append(success)

        s_r = np.mean(success_rate_per, axis=0)
        plt.bar(x + i*width, s_r, width=width, label=labels[i], tick_label = obj_name)
    # plt.xlabel()
    plt.title('success rate(after training %s iteration)' % (it*5))

    plt.legend()
    plt.savefig('/home/zjunesc/LQT/grasp/grasp_contactmap/grasp_envs/DAPG/exp_data/success rate(after training %s iteration).png' % (it*5), dpi=100)
    plt.show()
    # success = dict()
    # for k, v in log.items():
    #     if 'success' in k:
    #         success[k.split('_')[0]] = v
    #     if k == 'iteration':
    #         success[k] = v
    # df = pd.DataFrame(success)
    # bcr.bar_chart_race(df, 'test.mp4')
    # print('1')
# MAIN =========================================================
# Example: python make_train_plots.py --log_path logs/log.csv --keys eval_score rollout_score save_loc logs
def main():

    parent_log_dir_all = [
        # '../runs/curve/bottle_cap_resnet18',
                        '../runs/curve/bottle_cap_vt20t-reall-tmr05-bin-ft+dataset-BottleCap',
        '../runs/curve/bottle_cap_resnet18',
        '../runs/curve/bottle_cap_cnn',
                          # '../runs/curve/bottle_cap_v-repic-bin-ft-mr075-fr-mse+BottleCap',
                          # '../runs/curve/bottle_cap_vmvp',
                          #   '../runs/curve/bottle_cap_v-repic-t20-retac-ft-mr075-fr-mse-tmr05-bin-ft+BottleCap',
                          #   '../runs/curve/bottle_cap_t20-retac-tmr05-bin-ft+dataset-BottleCap',


                          ]
    log_dir_list_all = [
        ['seed333', 'seed444', 'seed666', 'seed888'],
        # ['seed111', 'seed222', 'seed333', 'seed444'],
        ['seed111', 'seed222', 'seed333', 'seed444'],#'seed888'
        # ['seed122', 'seed133', 'seed444', 'seed666'],
        ['seed111', 'seed222', 'seed333', 'seed444'],

    ]
    make_mean_std_plot(parent_log_dir_all=parent_log_dir_all,
                       log_dir_list_all=log_dir_list_all,
                       # success_rate/eval_success
                       key='Train/mean_success',
                       csv_name='log_train.csv')#log_objs_eval.csv
    # make_obj_plot(parent_log_dir_all=parent_log_dir_all,
    #               log_dir_list_all=log_dir_list_all,
    #               data=['grab', '3dnet'],
    #               metric='100_30')#log_objs_eval.csv
    # parent_log_dir_all = [
    #     #                   '/remote-home/share/lqt/grasp_contactmap14/grasp_envs/DAPG/TRO_all_pointnetL_pre_0801',
    # #                       '/remote-home/share/lqt/grasp_contactmap14/grasp_envs/DAPG/TRO_all_pointnetL_pre_0801',
    # #                       '/remote-home/share/lqt/grasp_contactmap14/grasp_envs/DAPG/TRO_all_pointnetL_pre_0801',
    #                       '/remote-home/share/lqt/grasp_contactmap14/grasp_envs/DAPG/TRO_all_pointnetL_pre_half0801',
    #                       '/remote-home/share/lqt/grasp_contactmap14/grasp_envs/DAPG/TRO_all_pointnetL_pre_half0801',
    #                       '/remote-home/share/lqt/grasp_contactmap14/grasp_envs/DAPG/TRO_all_pointnetL_pre_half0801'
    #                       ]
    # log_dir_list_all = [
    #     ['seed42bc150', 'seed123bc150', 'seed231bc150'],
    #     ['seed42bc100', 'seed123bc100', 'seed231bc100'],
    #     ['seed42bc125', 'seed123bc125', 'seed231bc125'],
    #     # ['seed42bc150', 'seed123bc150', 'seed231bc150'],
    #     # ['seed42bc175', 'seed123bc175', 'seed231bc175'],
    #     # ['seed42bc200', 'seed123bc200', 'seed231bc200'],
    # ]
    # make_mean_std_plot_one_exp(parent_log_dir_all=parent_log_dir_all,
    #                            log_dir_list_all=log_dir_list_all,
    #                            key='success_rate',
    #                            csv_name='log_train.csv')#log_objs_eval.csv

    # make_bar_plot(parent_log_dir_all=['/home/zjunesc/LQT/grasp/grasp_contactmap/grasp_envs/DAPG/exp_all1116',
    #                                        '/home/zjunesc/LQT/grasp/grasp_contactmap/grasp_envs/DAPG/exp_all_pointnet1116',
    #                                        '/home/zjunesc/LQT/grasp/grasp_contactmap/grasp_envs/DAPG/exp_all_pointnet1116_pre',
    #                                   '/home/zjunesc/LQT/grasp/grasp_contactmap/grasp_envs/DAPG/exp_all_pointnetG1116',
    #                                   '/home/zjunesc/LQT/grasp/grasp_contactmap/grasp_envs/DAPG/exp_all_pointnetG1116_pre',
    #                                   '/home/zjunesc/LQT/grasp/grasp_contactmap/grasp_envs/DAPG/exp_pointnet_pre1116',
    #                                   '/home/zjunesc/LQT/grasp/grasp_contactmap/grasp_envs/DAPG/exp_pointnetG1116',
    #                                   '/home/zjunesc/LQT/grasp/grasp_contactmap/grasp_envs/DAPG/exp_pointnetG1116_pre',
    #                                   '/home/zjunesc/LQT/grasp/grasp_contactmap/grasp_envs/DAPG/exp_pointnetG_pre1116'],
    #               log_dir_list_all = [['seed42', 'seed123'],
    #                                    ['seed42', 'seed123'],
    #                                    ['seed42', 'seed123'],
    #                                   ['seed_42', 'seed_123'],
    #                                   ['seed_42', 'seed_123'],
    #                                   ['seed_42', 'seed_123'],['seed_42', 'seed_123'],['seed_42', 'seed_123'],['seed_42', 'seed_123']],
    #               # success_rate/eval_success
    #               key='success_rate',
    #               csv_name='log_train.csv',
    #               it=59)
if __name__ == '__main__':
    main()



# import matplotlib.pyplot as plt
# import matplotlib.lines as lines
# import matplotlib.patches as mpatches
# # 自定义图例标记
# line1 = lines.Line2D([0], [0], label='DexRep', lw=2, c='#1f77b4')
# line2 = lines.Line2D([0], [0], label='DexRep+pGlo', lw=2, c='#ff7f0e')
# line3 = lines.Line2D([0], [0], label='Occ+Surf+pGlo', lw=2, c='#2ca02c')
# line4 = lines.Line2D([0], [0], label='Occ+Surf', lw=2, c='#d62728')
# line5 = lines.Line2D([0], [0], label='Surf', lw=2, c='#9467bd')
# line6 = lines.Line2D([0], [0], label='Loc-Geo', lw=2, c='#8c564b')
# line7 = lines.Line2D([0], [0], label='pGlo', lw=2, c='#e377c2')
# line8 = lines.Line2D([0], [0], label='Hand2obj', lw=2, c='#7f7f7f')
# # 构造图例
# # plt.figure() figsize: default: [6.4, 4.8]
# handles = [line1, line2, line3, line4, line5, line6, line7, line8]
# # 注意根据图例的行数调整figsize的高度（i.e., 0.32）
# fig, ax = plt.subplots(figsize=(15, 2))
# ax.legend(handles=handles, mode='expand', ncol=4, borderaxespad=0, fontsize=18, frameon=False)
# """
# legend常见参数设置
# borderpad:控制上下边距(default=0.4)
# borderaxespad:控制legend与图边框的距离(default=0.5)
# handlelength: handle与legend边框的距离(default=2)
# handletextpad: handle与text之间的距离(default=0.8)
# mode='expand', 水平展示图例
# frameon=False: 不要图例边框
# edgecolor: 图例边框颜色
# fontsize: 图例字体大小
# ncol=4    水平图例的个数
# """
# ax.axis('off')  # 去掉坐标的刻度
# plt.savefig('/home/zjunesc/LQT/grasp/grasp_contactmap/grasp_envs/DAPG/exp_data/label.png', dpi=500)
# plt.show()