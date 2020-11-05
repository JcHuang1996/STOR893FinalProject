import numpy as np
import os

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


#from https://tonysyu.github.io/plotting-error-bars.html#.WRwXWXmxjZs

def errorfill(x, y, color=None, alpha_fill=0.2, ax=None, pltcmd=None, linewidth = None, label=None):
    ax = ax if ax is not None else plt.gca()
    if color is None:
        color = ax._get_lines.color_cycle.next()

    ax.plot(x, y, pltcmd, color=color, linewidth = linewidth, label=label)


path = "./result/Swimmer-v1"

seed_list = [0]
seed_length = len(seed_list)

########################### Adam ############################################
adam = 0
for i in seed_list:
    adam += np.load(os.path.join(path, 'ADAM/adam_lr0.0001_seed%d_test_reward.npy' % i))
adam = adam/seed_length
adam_error = np.std(adam, axis=1)
adam = np.mean(adam, axis=1)

########################### SGD ############################################
sgd = 0
for i in seed_list:
    sgd += np.load(os.path.join(path, 'SGD/sgd_lr0.0001_seed%d_test_reward.npy' % i))
sgd = sgd/seed_length
sgd_error = np.std(sgd, axis=1)
sgd = np.mean(sgd, axis=1)

########################### ADAGRAD ############################################
adagrad = 0
for i in seed_list:
    adagrad += np.load(os.path.join(path, 'ADAGRAD/adagrad_lr0.0001_seed%d_test_reward.npy' % i))
adagrad = adagrad/seed_length
adagrad_error = np.std(adagrad, axis=1)
adagrad = np.mean(adagrad, axis=1)

########################### RMSProp ############################################
rmsprop = 0
for i in seed_list:
    rmsprop += np.load(os.path.join(path, 'RMSPROP/rmsprop_lr0.0001_seed%d_test_reward.npy' % i))
rmsprop = rmsprop/seed_length
rmsprop_error = np.std(rmsprop, axis=1)
rmsprop = np.mean(rmsprop, axis=1)


linewidth = 2
x = np.arange(70)

""" Figure of all methods in the appendix """
f = plt.figure(figsize=(9,7))

errorfill(label="Adam", x = x, y = adam,  color="r", pltcmd="-r", linewidth=linewidth)
        
errorfill(label="AdaGrad", x = x, y = adagrad,  color="b", pltcmd="-b", linewidth=linewidth)
        
errorfill(label="SGD", x = x, y = sgd,  color="c", pltcmd=":c", linewidth=linewidth)
        
errorfill(label="RMSProp", x = x, y = rmsprop,  color="m", pltcmd=":m", linewidth=linewidth)
        

        
fontsize = 14
plt.legend(prop={"size":fontsize}, loc="lower right", ncol=3) 
plt.xlabel('Time step', fontsize=fontsize)
plt.ylabel('Cumulative reward', fontsize=fontsize)
plt.xlim([-1, 70])
plt.ylim([-20, 50])
plt.grid()

fig_name = path + ".pdf"
f.savefig(fig_name, bbox_inches='tight')
fig_name = path + ".png"
f.savefig(fig_name, bbox_inches='tight')

plt.show()


