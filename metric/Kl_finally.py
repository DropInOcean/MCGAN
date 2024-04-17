import numpy as np
import scipy.stats
import scipy.io
import matplotlib.pyplot as plt

import os
import re
import random
import matplotlib.ticker as ticker

#Name of transfer value
Tr_value = 'T1_T2flair'
#Name of Generate value folder
Gv_folder = 'IXI'
#Name of GT value folder
# GT_folder = 'Simons'
#Name of save folder
save_folder = '/DATA2024/wsl/Metrics_final_show/'
hist_data_set1 = np.load(save_folder + 'Aucsf' + Gv_folder + '_' + Tr_value + '.npy')
hist_data_set2 = np.load(save_folder + 'pix2pixcsf' + Gv_folder + '_' + Tr_value + '.npy')
hist_data_set3 = np.load(save_folder + 'pGANcsf' + Gv_folder + '_' + Tr_value + '.npy')
hist_data_set4 = np.load(save_folder + 'cGANcsf' + Gv_folder + '_' + Tr_value + '.npy')
hist_data_set5 = np.load(save_folder + 'UNITcsf' + Gv_folder + '_' + Tr_value + '.npy')
hist_data_set6 = np.load(save_folder + 'Agcsf' + Gv_folder + '_' + Tr_value + '.npy')
hist_data_set_GT = np.load(save_folder + 'GTcsf' + Gv_folder + '_' + Tr_value + '.npy')


# x_axis = np.linspace(0,1,len(hist_data_set1))

plt.figure(figsize = (10,6))
x_axis = np.linspace(0,1,len(hist_data_set1))
#
plt.yticks([0.000000, 0.000030, 0.000060, 0.000090])
plt.tick_params(axis='both', labelsize =22, width = 4)
# plt.gca().yaxis.set_major_formatter(formatter)
# plt.ticklabel_format(style='sci', axis ='y', scilimits=(0,0))



plt.gca().yaxis.get_offset_text().set_fontsize(18)
plt.gca().yaxis.get_offset_text().set_weight('bold')


plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
ax = plt.gca()
for spine in ['top', 'right', 'bottom', 'left']:
    ax.spines[spine].set_linewidth(2)
plt.ticklabel_format(style='sci', axis ='y', scilimits=(0,0), useOffset=False)
# plt.gca().spines['top'].set_visible(False)
# plt.gca().spines['top'].set_visible(False)



plt.plot(x_axis, hist_data_set1/(256*256), color  = 'blue')
plt.plot(x_axis, hist_data_set2/(256*256), color = 'yellow')
plt.plot(x_axis, hist_data_set3/(256*256), color = 'green')
plt.plot(x_axis, hist_data_set4/(256*256), color = 'orange')
plt.plot(x_axis, hist_data_set5/(256*256), color = 'purple')
plt.plot(x_axis, hist_data_set6/(256*256), color = 'red', linewidth = 3)
plt.plot(x_axis, hist_data_set_GT/(256*256), color = 'black',linewidth = 5)
# ax.yaxis.set_major_locator(ticker.FixedLocator([ 0.000000, 0.33e-4, 0.66e-4, 0.99e-4, 1.2e-4]))


# 保存图像
plt.savefig(save_folder + Gv_folder + '_' + Tr_value + '_csf' + 'ad.png', bbox_inches = 'tight', pad_inches = 0)
# 显示图像
plt.show()


