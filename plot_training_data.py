import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D

mpl.rcParams['axes.linewidth'] = 2.5 # set the value globally
mpl.rcParams['xtick.major.size'] = 5.5 # set the value globally
mpl.rcParams['ytick.major.size'] = 5.5 # set the value globally

tick_fontsize = 18
title_fontsize = 27
axis_label_fontsize=24
legend_fontsize = 17

SAVE_FIGS = False
cyto_names = ["TNF", "sTNFr", "IL10", "GCSF", "IFNg", "PAF", "IL1", "IL4", "IL8", "IL12", "sIL1r"]
# TNF - inflam
# sTNFr - inhib
# IL10 -inhib
# GCSF - inflam

heal_color = "green"
timeout_color = "red"
death_color = "red"
infection_color = "black"

lower_data_bound = 0000
upper_data_bound = 4000
print("looking at data from " + str(lower_data_bound) + " to " + str(upper_data_bound))
full_data = np.load("./IIRABM_DRL_Experiments/limited_control/0_cyto_test0.npy")
full_data = full_data[:,:,:4199]
full_data[full_data==0] = np.nan
print(full_data.shape)
full_data = np.moveaxis(full_data, 0,-1)
cytokine_data = full_data[[0,2,3,4,5,12,13,14,15,16,17,18],:,lower_data_bound:upper_data_bound]
infection_data = full_data[1,:,lower_data_bound:upper_data_bound]
pmn_data = full_data[8,:,lower_data_bound:upper_data_bound]
mono_data = full_data[9,:,lower_data_bound:upper_data_bound]
th1_data = full_data[10,:,lower_data_bound:upper_data_bound]
th2_data = full_data[11,:,lower_data_bound:upper_data_bound]
print("Max infection level: " + str(np.nanmax(infection_data)))
action_data = np.load("./IIRABM_DRL_Experiments/limited_control/0_action_test0.npy")
action_data = np.moveaxis(action_data, 0,-1)
action_data = action_data[:,:,:4199]

action_data = action_data[:,:,lower_data_bound:upper_data_bound]
print("Cytokine data shape: " + str(cytokine_data.shape))

for i in range(action_data.shape[2]):
    for j in range(action_data.shape[1]):
        if np.all(action_data[:,j,i] == action_data[0,j,i]):
            action_data[:,j,i] = np.nan

heal_index = np.zeros(cytokine_data.shape[2])
for i in range(len(heal_index)):
    if np.any(cytokine_data[0,500:,i] < 600):
        heal_index[i] = 1
    if np.any(cytokine_data[0,100:,i] > 8000):
        heal_index[i] = 2

summed_cytokine_data = np.sum(cytokine_data[1:,:,:], axis=0)
summed_action_data = np.sum(action_data, axis=0)

summed_healed_cyto = summed_cytokine_data[:,heal_index == 1]
summed_dead_cyto = summed_cytokine_data[:,heal_index == 2]
summed_timeout_cyto = summed_cytokine_data[:,heal_index == 0]

healed_cyto = cytokine_data[:,:,heal_index == 1]
healed_action = action_data[:,:,heal_index == 1]
healed_inf = infection_data[:,heal_index == 1]
healed_pmn = pmn_data[:,heal_index == 1]
healed_mono = mono_data[:,heal_index == 1]
healed_TH1 = th1_data[:,heal_index == 1]
healed_TH2 = th2_data[:,heal_index == 1]

dead_cyto = cytokine_data[:,:,heal_index == 2]
dead_action = action_data[:,:,heal_index == 2]
dead_inf = infection_data[:,heal_index == 2]
dead_pmn = pmn_data[:,heal_index == 2]
dead_mono = mono_data[:,heal_index == 2]
dead_TH1 = th1_data[:,heal_index == 2]
dead_TH2 = th2_data[:,heal_index == 2]

timeout_cyto = cytokine_data[:,:,heal_index == 0]
timeout_action = action_data[:,:,heal_index == 0]
timeout_inf = infection_data[:,heal_index == 0]
timeout_pmn = pmn_data[:,heal_index == 0]
timeout_mono = mono_data[:,heal_index == 0]
timeout_TH1 = th1_data[:,heal_index == 0]
timeout_TH2 = th2_data[:,heal_index == 0]

print("Num Healed: " + str(summed_healed_cyto.shape[-1]))
print("Num Timeout " + str(summed_timeout_cyto.shape[-1]))
print("Num Dead " + str(summed_dead_cyto.shape[-1]))
print("data Categorized")
def plot_cytokine_action():
    index = 0
    for name in cyto_names:
        plt.figure(figsize=(10,6))
        try:
            plt.scatter(timeout_cyto[index+1,:,:], timeout_action[index,:,:], c=timeout_color, s=1, label="Timeout")
        except:
            pass
        try:
            plt.scatter(dead_cyto[index+1,:,:], dead_action[index,:,:], c=death_color, s=1, label="Dead")
        except:
            pass
        try:
            plt.scatter(healed_cyto[index+1,:,:], healed_action[index,:,:], c=heal_color, s=1, label="Heal")
        except:
            pass
        plt.title("Cytokine Total vs Action Magnitude for " + name)
        plt.xlabel("Cytokine Total")
        plt.ylabel("Action Magnitude")
        plt.ylim([0.0005,120])
        plt.yscale("log")
        plt.legend(loc=4)
        if SAVE_FIGS:
            plt.savefig("./Plots/"+ name +"_cyto.png")
        index += 1
#
def plot_cytokines(plot_live=True, plot_timeout=True, plot_dead=True, plot_inf=True):
    if plot_timeout:
        try:
            plt.plot(summed_timeout_cyto[:,0], c=timeout_color, label="Non-Healing")
            plt.plot(summed_timeout_cyto, c=timeout_color)
            if plot_inf:
                plt.plot(timeout_inf, c=infection_color)
        except:
            pass
    if plot_dead:
        try:
            plt.plot(summed_dead_cyto[:,0], c=death_color, label="Dead")
            plt.plot(summed_dead_cyto, c=death_color)
            if plot_inf:
                plt.plot(dead_inf, c=infection_color)
        except:
            pass
    if plot_live:
        try:
            plt.plot(summed_healed_cyto[:,0], c=heal_color, label="Heal")
            plt.plot(summed_healed_cyto, c=heal_color)
            if plot_inf:
                plt.plot(healed_inf, c=infection_color)
        except:
            pass
    if plot_inf:
        plt.plot(infection_data[0,0], c=infection_color, label="Infection")
    # plt.plot(infection_data, c=infection_color)

    plt.title("Cytokine Total over Time", fontsize=title_fontsize)
    plt.xlabel("Step (6 min)", fontsize=axis_label_fontsize)
    plt.ylabel("Cyotkine Total", fontsize=axis_label_fontsize)

    for tick in plt.gca().xaxis.get_major_ticks():
        tick.label1.set_fontsize(tick_fontsize)
        # tick.label1.set_fontweight('bold')
    for tick in plt.gca().yaxis.get_major_ticks():
        tick.label1.set_fontsize(tick_fontsize)
        # tick.label1.set_fontweight('bold')
    legend_elements = [Line2D([0], [0], color=heal_color, lw=4, label='Healing'),
                        Line2D([0], [0], color=timeout_color, lw=4, label='Non-Healing')]
    plt.legend(handles=legend_elements, bbox_to_anchor =(1.12, 1.175), prop={'size':legend_fontsize})
    if SAVE_FIGS:
        plt.savefig("./Plots/Cytokine_total_time.png")

def plot_cytokine_action_time():
    steps = np.arange(timeout_action.shape[1])
    timeout_steps = steps
    dead_steps = steps
    healed_steps = steps
    for i in range(timeout_action.shape[2]-1):
        timeout_steps = np.vstack((timeout_steps, steps))
    for i in range(dead_action.shape[2]-1):
        dead_steps = np.vstack((dead_steps,steps))
    for i in range(healed_action.shape[2]-1):
        healed_steps = np.vstack((healed_steps, steps))
    index = 0
    for name in cyto_names:
        plt.figure(figsize=(10,6))
        plt.subplot(211)
        try:
            plt.scatter(timeout_steps[:,100:], timeout_action[index,100:,:].T, c=timeout_color, label="Timeout", s=.7)
        except:
            pass
        try:
            plt.scatter(dead_steps.T, dead_action[index,:,:], c=death_color, label="Dead", s=.7)
        except:
            pass
        try:
            plt.scatter(healed_steps[:,100:], healed_action[index,100:,:].T, c=heal_color, label="Heal", s=.7)
        except:
            pass
        plt.title("Action Magnitude of "+ name +" over Time")
        plt.xlabel("Step (6 min)", fontsize=axis_label_fontsize)
        plt.ylabel("Action Magnitude")
        plt.ylim([0.0005,120])
        plt.yscale("log")
        plt.legend(loc=4)

        plt.subplot(212)
        plot_cytokines()

        if SAVE_FIGS:
            plt.savefig("./Plots/"+ name +"_time.png")
        index += 1
    plt.figure(figsize=(10,6))
    plot_cytokines()

def plot_compare_cytokine():
    index = 0
    for name in cyto_names:
        fig = plt.figure(figsize=(10,6))
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2, sharey=ax1)
        try:
            ax1.plot(healed_cyto[index+1,:,0], c="g", label="Heal")
            ax1.plot(healed_cyto[index+1,:,:], c="g")
        except:
            pass
        ax1.set_xlabel("Step (6 min)", fontsize=axis_label_fontsize)
        ax1.set_ylabel("Cyotkine Count", fontsize=axis_label_fontsize)
        ax1.legend(loc=4)
        try:
            ax2.plot(timeout_cyto[index+1,:,0], c="m", label="Timeout")
            ax2.plot(timeout_cyto[index+1,:,:], c="m")
        except:
            pass
        ax2.set_xlabel("Step (6 min)", fontsize=axis_label_fontsize)
        ax2.set_ylabel("Cyotkine Count", fontsize=axis_label_fontsize)
        ax2.legend(loc=4)

        ax1.set_title(name+" Count over Time")
        if SAVE_FIGS:
            fig.savefig("./Plots/"+name+"_count")
        index += 1

def plot_actions_infection_cytos(plot_live=True, plot_timeout=True, plot_dead=True, plot_inf=True, seperate_plots=False):
    steps = np.arange(timeout_action.shape[1])
    timeout_steps = steps
    dead_steps = steps
    healed_steps = steps
    for i in range(timeout_action.shape[2]-1):
        timeout_steps = np.vstack((timeout_steps, steps))
    for i in range(dead_action.shape[2]-1):
        dead_steps = np.vstack((dead_steps,steps))
    for i in range(healed_action.shape[2]-1):
        healed_steps = np.vstack((healed_steps, steps))
    index = 0

    for name in cyto_names:
        if seperate_plots:
            ax1 = plt.figure(figsize=(10,7)).gca()
            ax2 = plt.figure(figsize=(10,7)).gca()
            ax3 = plt.figure(figsize=(10,7)).gca()
        else:
            fig = plt.figure(figsize=(10,8))
            ax1 = fig.add_subplot(3,1,1)
            ax2 = fig.add_subplot(3,1,2)
            ax3 = fig.add_subplot(3,1,3)
        if plot_timeout:
            try:
                ax1.scatter(timeout_steps[:,100:], timeout_action[index,100:,:].T, c=timeout_color, label="Non-Healing", s=.7)
            except:
                pass
        if plot_dead:
            try:
                ax1.scatter(dead_steps[:,100:], dead_action[index,100:,:].T, c=death_color, label="Dead", s=.7)
            except:
                pass
        if plot_live:
            try:
                ax1.scatter(healed_steps[:,100:], healed_action[index,100:,:].T, c=heal_color, label="Heal", s=.7)
            except:
                pass

        for tick in ax1.xaxis.get_major_ticks():
            tick.label1.set_fontsize(tick_fontsize)
            # tick.label1.set_fontweight('bold')
        for tick in ax1.yaxis.get_major_ticks():
            tick.label1.set_fontsize(tick_fontsize)
            # tick.label1.set_fontweight('bold')
        ax1.set_title("Action Magnitude of "+ name +" over Time", fontsize=18)
        ax1.set_xlabel("Step (6 min)", fontsize=axis_label_fontsize)
        ax1.set_ylabel("Action Magnitude", fontsize=axis_label_fontsize)
        ax1.set_ylim([0.0005,120])
        ax1.set_yscale("log")

        legend_elements = [Line2D([0], [0], color=heal_color, lw=4, label='Healing'),
                            Line2D([0], [0], color=timeout_color, lw=4, label='Non-Healing')]

        ax1.legend(handles=legend_elements, bbox_to_anchor =(1.12, 1.175), prop={'size':legend_fontsize})
        index += 1

        if not seperate_plots:
            plt.subplot(313, sharex=ax1)
        else:
            ax3.sharex(ax1)
            plt.sca(ax3)
        plot_cytokines(plot_live, plot_timeout, plot_dead, plot_inf)


        if not seperate_plots:
            plt.subplot(313, sharex=ax1)
        else:
            ax2.sharex(ax1)
            plt.sca(ax2)
        ax2.set_yscale("linear")
        if plot_live:
            try:
                plt.plot(healed_inf,c=infection_color)
            except:
                pass
        if plot_dead:
            try:
                plt.plot(dead_inf, c=infection_color)
            except:
                pass
        if plot_timeout:
            try:
                plt.plot(timeout_inf, c=infection_color)
            except:
                pass
        for tick in ax2.xaxis.get_major_ticks():
            tick.label1.set_fontsize(tick_fontsize)
            # tick.label1.set_fontweight('bold')
        for tick in ax2.yaxis.get_major_ticks():
            tick.label1.set_fontsize(tick_fontsize)
            # tick.label1.set_fontweight('bold')
        plt.plot(infection_data[0,0], c=infection_color, label="infection")
        plt.title("Infection over time", fontsize=title_fontsize)
        plt.xlabel("Step (6 min)", fontsize=axis_label_fontsize)
        plt.ylabel("Infection Count", fontsize=axis_label_fontsize)

        plt.legend()

        if SAVE_FIGS:
            if seperate_plots:
                ax1.get_figure().savefig("./Plots/"+name+"_action_nonheal")
                ax2.get_figure().savefig("./Plots/Infection_nonheal")
                ax3.get_figure().savefig("./Plots/Cytokines_nonheal")
            else:
                fig.savefig("./Plots/"+name+"_action_infection_cyto")


def plot_infection_cyokine():
    fig = plt.figure(figsize=(10,6))

    ax1 = plt.subplot(212)
    plot_cytokines(plot_live=False)
    plt.subplot(211, sharey=ax1)
    plot_cytokines(plot_timeout=False)

    if SAVE_FIGS:
        fig.savefig("./Plots/infection_cyto")

def plot_action_cell_count(plot_live=True, plot_timeout=True, plot_dead=True):
        steps = np.arange(timeout_action.shape[1])
        timeout_steps = steps
        dead_steps = steps
        healed_steps = steps
        for i in range(timeout_action.shape[2]-1):
            timeout_steps = np.vstack((timeout_steps, steps))
        for i in range(dead_action.shape[2]-1):
            dead_steps = np.vstack((dead_steps,steps))
        for i in range(healed_action.shape[2]-1):
            healed_steps = np.vstack((healed_steps, steps))
        index = 0

        for name in cyto_names:
            fig = plt.figure(figsize=(10,8))
            # ax1 = fig.add_subplot(3,1,1)
            # ax2 = fig.add_subplot(3,1,2)
            ax1 = plt.subplot(511)
            if plot_timeout:
                try:
                    ax1.scatter(timeout_steps[:,100:], timeout_action[index,100:,:].T, c=timeout_color, label="Timeout", s=.7)
                except:
                    pass
            if plot_dead:
                try:
                    ax1.scatter(dead_steps.T, dead_action[index,:,:], c=death_color, label="Dead", s=.7)
                except:
                    pass
            if plot_live:
                try:
                    ax1.scatter(healed_steps[:,100:], healed_action[index,100:,:].T, c=heal_color, label="Heal", s=.7)
                except:
                    pass
            ax1.tick_params(axis='x', bottom=False, labelbottom=False)
            plt.title("Action Magnitude of "+ name +" over Time")
            plt.xlabel("Step (6 min)")
            plt.ylabel("Action Magnitude")
            plt.ylim([0.0005,120])
            plt.yscale("log")
            plt.legend(loc=4)
            index += 1
            ax2 = plt.subplot(512, sharex=ax1)
            ax2.tick_params(axis='x', bottom=False, labelbottom=False)
            plt.title("PMN array total")
            ax3 = plt.subplot(513)
            ax3.tick_params(axis='x', bottom=False, labelbottom=False)
            plt.title("Mono array total")
            ax4 = plt.subplot(514)
            ax4.tick_params(axis='x', bottom=False, labelbottom=False)
            plt.title("TH1 array total")
            ax5 = plt.subplot(515)
            plt.title("TH2 array total")

            if plot_timeout:
                try:
                    ax2.plot(timeout_pmn, c=timeout_color, label="Timeout")
                    ax3.plot(timeout_mono,c=timeout_color, label="Timeout")
                    ax4.plot(timeout_TH1,c=timeout_color, label="Timeout")
                    ax5.plot(timeout_TH2,c=timeout_color, label="Timeout")
                except:
                    pass
            if plot_dead:
                try:
                    ax2.plot(dead_pmn, c=death_color, label="Dead")
                    ax3.plot(dead_mono, c=death_color, label="Dead")
                    ax4.plot(dead_TH1, c=death_color, label="Dead")
                    ax5.plot(dead_TH2, c=death_color, label="Dead")
                except:
                    pass
            if plot_live:
                try:
                    ax2.plot(healed_pmn, c=heal_color, label="Healed")
                    ax3.plot(healed_mono, c=heal_color, label="Healed")
                    ax4.plot(healed_TH1, c=heal_color, label="Healed")
                    ax5.plot(healed_TH2, c=heal_color, label="Healed")
                except:
                    pass
            if SAVE_FIGS:
                plt.savefig("./Plots/"+name+"_action_cell_count")


def plot_oxydef():
    plt.figure(figsize=(10,7))
    try:
        plt.plot(healed_cyto[0,:,0],c=heal_color, label="Healed")
        plt.plot(healed_cyto[0,:,:], c=heal_color)
    except:
        pass
    try:
        plt.plot(dead_cyto[0,:,0], c=death_color, label="Dead")
        plt.plot(dead_cyto[0,:,:], c=death_color)
    except:
        pass
    try:
        plt.plot(timeout_cyto[0,:,0], c=timeout_color, label="Timeout")
        plt.plot(timeout_cyto[0,:,:], c=timeout_color)
    except:
        pass
    legend_elements = [Line2D([0], [0], color=heal_color, lw=4, label='Healing'),
                        Line2D([0], [0], color=timeout_color, lw=4, label='Non-Healing')]

    plt.legend(handles=legend_elements, bbox_to_anchor =(1.12, 1.175), prop={'size':legend_fontsize})
    plt.title("Oxygen Deficit", fontsize=title_fontsize)
    plt.xlabel("Step (6min)", fontsize=axis_label_fontsize)
    plt.ylabel("Oxygen Deficit", fontsize=axis_label_fontsize)

    for tick in plt.gca().xaxis.get_major_ticks():
        tick.label1.set_fontsize(tick_fontsize)
        # tick.label1.set_fontweight('bold')
    for tick in plt.gca().yaxis.get_major_ticks():
        tick.label1.set_fontsize(tick_fontsize)
        # tick.label1.set_fontweight('bold')
    if SAVE_FIGS:
        plt.savefig("./Plots/Oxy_Def_nonheal")
# plot_cytokine_action()
# plt.close('all')
# plot_cytokine_action_time()
# plt.close('all')
# plot_compare_cytokine()
# plt.close('all')
plot_actions_infection_cytos(plot_timeout=True, plot_live=True, plot_dead=False, seperate_plots=True, plot_inf=False)
# plt.close('all')
# plot_infection_cyokine()
# plt.close('all')
# plot_action_cell_count(plot_live=True)
# plt.close('all')
# plot_cytokines()
# plt.close('all')
plot_oxydef()
plt.show()
