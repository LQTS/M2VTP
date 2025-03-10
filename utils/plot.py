import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt

# train = [0.97,0.81,0.74,0.82,0.33,]
# train_error = [0.03,0.08,0.15,0.21,0.04]
#
# eval = [0.97,0.74,0.59,0.85,0.23]
# eval_error = [0.04,0.06,0.29,0.1,0.05]
# train = [0.97,0.3,0.12]
# train_error = [0.03,0.27,0.06]
#
# eval = [0.97,0.32,0.13]
# eval_error = [0.04,0.41,0.05]

size = 3
x = np.arange(size)
a = [97,30,12]
a_SD = [3,27,6]
b = [97,32,13]
b_SD = [4,11,5]

total_width, n = 0.8, 2
width = total_width / n
x = x - (total_width - width) / 2
# labels = ['Ours', 'V-Only', 'MVP', 'VT-Sep', 'T-Only']
labels = ['Ours', 'VT-Scr-R', 'VT-Scr-C']
# plt.title('Evaluation', fontsize=20)
plt.bar(x, a,  width=width, yerr = a_SD, tick_label=labels, label='seen')
plt.bar(x + width, b, width=width, yerr = b_SD, tick_label=labels ,label='unseen')
plt.ylabel('Success Rate(%)', fontsize=18)
plt.xlabel('Method', fontsize=18)
plt.legend()
plt.savefig('../runs/curve/plot/exp1_evaluation_SR.png', dpi=1000)
plt.show()