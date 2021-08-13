import numpy as np
import pandas as pd
import random
from matplotlib import pyplot as plt
import matplotlib.cm as matplotlibcm
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
import matplotlib.patches as mpatches
import seaborn as sns
import os
import time
import pdb
from utilities import *

import io_dict_to_hdf5 as ioh5

color_names=['windows blue',
             'red',
             'amber',
             'faded green',
             'dusty purple',
             'orange',
             'steel blue',
             'pink',
             'greyish',
             'mint',
             'clay',
             'light cyan',
             'forest green',
             'pastel purple',
             'salmon',
             'dark brown',
             'lavender',
             'pale green',
             'dark red',
             'gold',
             'dark teal',
             'rust',
             'fuchsia',
             'pale orange',
             'cobalt blue',
             'mahogany',
             'cloudy blue',
             'dark pastel green',
             'dust',
             'electric lime',
             'fresh green',
             'light eggplant',
             'nasty green']
 
color_palette = sns.xkcd_palette(color_names)
colors = sns.xkcd_palette(color_names)
cc = sns.xkcd_palette(color_names)
sns.set_style("darkgrid")
sns.set_context("notebook")

Plot_Dir = './plots/'

Arena_Length = 1104 - 80

plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelweight']='bold'
plt.rcParams['figure.titleweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
#-------------------------------------------------------------------------------
# Dictionaries for decoding trial summary data
Cond_Dict = {0:'100-0', 1:'80-20', 2:'60-40', 3:'Control', 4:'1% Abs-Conc',5:'0.1% Abs-Conc'}
Port_Dict = {2:'Left', 1:'Right'}
Resp_Dict = {1:'Correct',0:'Incorrect'}
Turn_Dict = {2:'Left', 1:'Right'}
Cond_InvDict = {'100-0':0, '80-20':1, '60-40':2, 'Control':3, '1% Abs-Conc':4, '0.1% Abs-Conc':5}
Port_InvDict = {'Left':2, 'Right':1}
Resp_InvDict = {'Correct':1,'Incorrect':0}


#-------------------------------------------------------------------------------
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

#-------------------------------------------------------------------------------
def gradient_cmap(gcolors, nsteps=256, bounds=None):

    ncolors = len(gcolors)
    if bounds is None:
        bounds = np.linspace(0, 1, ncolors)

    reds = []
    greens = []
    blues = []
    alphas = []
    for b, c in zip(bounds, gcolors):
        reds.append((b, c[0], c[0]))
        greens.append((b, c[1], c[1]))
        blues.append((b, c[2], c[2]))
        alphas.append((b, c[3], c[3]) if len(c) == 4 else (b, 1., 1.))

    cdict = {'red': tuple(reds),
             'green': tuple(greens),
             'blue': tuple(blues),
             'alpha': tuple(alphas)}

    cmap = LinearSegmentedColormap('grad_colormap', cdict, nsteps)
    return cmap

#-------------------------------------------------------------------------------
def shuffle_colors(used_states):
    if isinstance(used_states,list):
        mask = np.zeros(len(colors), dtype=bool)
        mask[used_states] = True

        color_names_shortened = [cn for cn, s in zip(color_names,mask) if s]
        color_names_shuffled = [x for _, x in sorted(zip(used_states,color_names_shortened), key=lambda pair: pair[0])]
        colors_shf = sns.xkcd_palette(color_names_shuffled)
        cc = [rgb for rgb in colors_shf]
    else:
        K = used_states
        cc, colors_shf = get_colors(K)

    return cc,colors_shf

#-------------------------------------------------------------------------------
def get_colors(N_used_states):
    names = color_names[:N_used_states]
    colors_out = sns.xkcd_palette(names)
    cc = [rgb for rgb in colors_out]

    return cc, colors_out


#-------------------------------------------------------------------------------
def plot_z_samples(zs,used_states,xmax=None,
                   plt_slice=None, N_iters=None,
                   title=None, ax=None, pdf=None):
    if ax is None:
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111)

    zs = np.array(zs)
    if plt_slice is None:
        plt_slice = (0, zs.shape[1])
    if N_iters is None:
        N_iters = zs.shape[0]
    if isinstance(used_states,list) or isinstance(used_states,np.array):
        #cc = [rgb for rgb, s in zip(colors.colors,mask) if s]
        cc, colors_shf = shuffle_colors(used_states)
        K = len(used_states)     # number of discovered states?
    else:
        K = used_states
        cc, colors_shf = get_colors(K)
        used_states = range(K)

    # Plot StateSeq as a heatmap
    if zs.ndim==2:
        im = ax.imshow(zs[:, slice(*plt_slice)], aspect='auto', vmin=0, vmax=K - 1,
                       cmap=gradient_cmap(cc), interpolation="nearest",
                       extent=plt_slice + (N_iters, 0))
    elif zs.ndim==3:
#        current_cmap = matplotlibcm.get_cmap()
#        current_cmap.set_bad('w',1.)#(color='white')
        im = ax.imshow(zs, aspect='auto', extent=plt_slice + (N_iters, 0))#, cmap=current_cmap)

    # Create a legend
    # get the colors of the values, according to the colormap used by imshow
    # create a patch (proxy artist) for every color
    patches = [mpatches.Patch(color=colors_shf[s], label="State {:.0f}".format(s)) for s in used_states]
    # put those patched as legend-handles into the legend
    plt.legend(handles=patches, loc=4)

    ax.set_ylabel("Trial")
    ax.set_xlabel("Frame #")
    ax.grid(False)
    if xmax is not None:
        plt.xlim(0,xmax)

    if title is not None:
        ax.set_title(title)

    if pdf is not None:
        pdf.savefig(fig)
        plt.close('all')

#-------------------------------------------------------------------------------
def plot_MAPorMean_estimates_in_cond(trMAPs,used_states,trialsummary,mouseID, verbose=True,
                             cond_port_resp_turn=None, MAP_times=None,pdfdoc=None,MEAN=False):
    """Plot the discrete state sequence for the collection of trials matching
       conditions encoded in cond_port_resp_turn
    """
    if cond_port_resp_turn is None:
        mask = True
    else:
        # Find the trials that correspond to the conditions
        cond = cond_port_resp_turn[0] #  100-0, 80-20, 60-40, control, Abs... Abs...
        port = cond_port_resp_turn[1] # location of reward port {L: 2, R: 1}
        resp = cond_port_resp_turn[2] # correct (1) or incorrect (0) mouse response
        turn = cond_port_resp_turn[3] if len(cond_port_resp_turn)>3 else None # direction of mouse's initial turn {L: 2, R: 1}
        condi = True if cond is None else np.isin(trialsummary['cond'], cond)
        porti = True if port is None else np.isin(trialsummary['port'], port)
        respi = True if resp is None else np.isin(trialsummary['resp'], resp)
        turni = True if turn is None else np.isin(trialsummary['turn'], turn)
        mask =  porti & condi & respi & turni    # = (trialsummary['port'] == lr) & (trialsummary['cond'] == iCond) & (trialsummary['resp'] == resp)
        if np.sum(mask) == 0:
            if verbose:
                print('No matching conditions found.')
            return

    if isinstance(trMAPs,list): #does not assume sorted trMAPs (dec. order of trial duration)
        assert len(trialsummary)==len(trMAPs)
        # Create a new list based of only selected conditions
        if mask is True:
            cond_MAPs = trMAPs
        else:
            cond_MAPs = [trMAPs[i] for i in range(len(trMAPs)) if mask[i]]
        trial_times = np.asarray([len(MAP) for MAP in cond_MAPs])
        MAPorder = np.argsort(trial_times)[::-1]
        trial_times = trial_times[MAPorder]
        #cond_MAPs.sort(key=len,reverse=True)

        # Create a np.ndarray of MAPs with padded NaNs to look at all trials of selected conditions in a heatmap
        max_tsteps = trial_times[0] #len(cond_MAPs[0])
        if not MEAN: #put MAP estiamtes in MAP_padded
            MAP_padded = np.empty((len(cond_MAPs),max_tsteps)) #((sum(mask),max_tsteps))
            MAP_padded[:] = np.nan
            for tr, MAP in enumerate(cond_MAPs):
                MAP_padded[tr, :len(MAP)] = MAP
        else: # put colors corresponding to Posterior mean in MAP_padded
            MAP_padded = np.empty((len(cond_MAPs),max_tsteps,3)) #last dimension is for color RGB
            MAP_padded[:] = .9 #np.nan
            cc, _ = shuffle_colors(used_states)
            cc = np.asarray(cc)
            for tr, MAP in enumerate(cond_MAPs):
#                pdb.set_trace()
                MAP_padded[tr, :len(MAP),:] = MAP @ cc  # MAP.shape = (trial_length, num_states), cc.shape = (num_states,3)
        MAP_padded = MAP_padded[MAPorder]
    elif isinstance(trMAPs,np.ndarray): #does assume sorted trMAPs, trialsummary and MAP_times
        assert MAP_times is not None
        assert len(trialsummary)==len(trMAPs)==len(MAP_times)
        MAP_padded = trMAPs[mask]
        trial_times = MAP_times[mask]
        #max_tsteps = (~np.isnan(MAP_padded)).nonzero()[1].max()
        max_tsteps = trial_times[0]#assume trial_times is sorted in decreasing order

    xmax = np.minimum(max_tsteps, int(50*round(1.5*np.median(trial_times)/50)) )

    if cond_port_resp_turn is None:
        title = 'MAP estimates for all {} trials of {}'.format(MAP_padded.shape[0],mouseID)
    elif type(cond)==type(port)==type(resp)==int:
        title = '{} Condition, MAP estimates for {} {} trials of the {} Active Odor Port for {}'.format(Cond_Dict[cond],MAP_padded.shape[0],Resp_Dict[resp],Port_Dict[port],mouseID)
        #  title = 'MAP estimates for {} {}, {} trials of the {} Active Odor Port'.format(sum(mask),Cond_Dict[iCond],Resp_Dict[Port_Dict[lr])
        if type(turn)==int:
            title = '{} Condition, MAP estimates for {} {} trials of the {} Port and {} Turn for {}'.format(Cond_Dict[cond],MAP_padded.shape[0],Resp_Dict[resp],Port_Dict[port],Turn_Dict[turn],mouseID)
    else:
        title = 'MAP estimates for {} trials of specified conditions for {}'.format(MAP_padded.shape[0],mouseID)

    plot_z_samples(MAP_padded, used_states,xmax, title= title, pdf= pdfdoc)
    if cond_port_resp_turn is None:      #isinstance(trMAPs,list):
        return MAPorder, MAP_padded, trial_times


#-------------------------------------------------------------------------------
def plot_MAP_estimates(trMAPs,trMASKs,used_states,trialsummary,args,mouseID,Plot_Dir=None,fname=None, SAVEFIG=False):
    if SAVEFIG:
        timestr = time.strftime('%Y%m%d_%H%M')
        if Plot_Dir is None:
            Plot_Dir = './plots'

        # Create a pdf document for plots
        if fname is None:
            # Create a pdf document for plots
            if args['model'] is 'AR-HDP-HMM':
                fname = 'MAP_StateSeq_{}_K{:.0f}G{:.0f}_{}.pdf'.format(mouseID,args['K'],args['G'],timestr)
            else:
                fname = 'MAP_StateSeq_{}_N{}_{}.pdf'.format(mouseID,args['Nmax'],timestr)
        pdfdoc = PdfPages(os.path.join(Plot_Dir,fname))
    else:
        pdfdoc = None

    #Apply mask to MAP state sequence
    for MAP,mask in zip(trMAPs,trMASKs):
        MAP[mask] = np.nan

    # Plot ARHMM MAP State Sequences in all trials:
    MAPorder, MAP_padded, trial_times = plot_MAPorMean_estimates_in_cond(
                            trMAPs,used_states,trialsummary,mouseID,pdfdoc=pdfdoc)

    #trialsummary = trialsummary.iloc[MAPorder] #sorts trialsummary in dec. order of trial length

    # Plot ARHMM MAP State Sequences in different conditions:
    for resp in Resp_Dict: #correct or incorrect trials
        for iCond in Cond_Dict: #different task conditions
            for lr in Port_Dict: #left or right reward/odor port
                for turn in Turn_Dict: #left or right intial turn
                    #mask =
                    plot_MAPorMean_estimates_in_cond(trMAPs,  #MAP_padded,
                        used_states,trialsummary,mouseID,verbose=False,pdfdoc=pdfdoc,
                        cond_port_resp_turn=(iCond,lr,resp,turn), MAP_times=trial_times)
    if SAVEFIG:
        pdfdoc.close()
        plt.close('all')
        
#-------------------------------------------------------------------------------
def old_plot_MAP_estimates(trMAPs,trMASKs,used_states,trialsummary,args,mouseID,Plot_Dir=None,fname=None, SAVEFIG=False):
    pass
    # # Get number of trials from trial summary data frame
    # timestr = time.strftime('%Y%m%d_%H%M')
    # if SAVEFIG:
    #     if Plot_Dir is None:
    #         Plot_Dir = './plots'
    #
    #     # Create a pdf document for plots
    #     if fname is None:
    #         # Create a pdf document for plots
    #         if args['model'] is 'AR-HDP-HMM':
    #             fname = 'MAP_StateSeq_{}_K{:.0f}G{:.0f}_{}.pdf'.format(mouseID,args['K'],args['G'],timestr)
    #         else:
    #             fname = 'MAP_StateSeq_{}_N{}_{}.pdf'.format(mouseID,args['Nmax'],timestr)
    #     pdfdoc = PdfPages(os.path.join(Plot_Dir,fname))
    # else:
    #     pdfdoc = None
    #
    # #Apply mask to MAP state sequence
    # for MAP,mask in zip(trMAPs,trMASKs):
    #     MAP[mask] = np.nan
    #
    # #Plot all MAP sequences on one plot
    # trMAPs_sorted = sorted(trMAPs,key=len,reverse=True)
    # max_tsteps = len(trMAPs_sorted[0])
    #
    # nTrials = len(trMAPs) #len(trialsummary)
    # MAP_padded = np.empty((nTrials,max_tsteps))
    # MAP_padded[:] = np.nan
    # for ii, MAP in enumerate(trMAPs_sorted):
    #     #pdb.set_trace()
    #     MAP_padded[ii,:len(MAP)] = MAP
    #
    # # Get a list of trial lengths
    # MAP_ts = [len(trMAPs[i]) for i in range(nTrials)]
    # MAP_ts.sort(key=int,reverse=True)
    # xmax = np.minimum(max_tsteps, int(50*round(1.5*np.median(MAP_ts)/50)) )
    #
    # # Plot the discrete state sequence of the collection of trials
    # title = 'MAP estimates for all {} trials of {}'.format(len(trMAPs),mouseID)
    # plot_z_samples(MAP_padded,used_states,xmax,title = title,pdf = pdfdoc)
    #
    # # Plot ARHMM MAP State Sequences
    # for resp in Resp_Dict:
    #      # Loop over conditions
    #     for iCond in Cond_Dict:
    # #         # Loop over active odor porth
    #         for lr in Port_Dict:
    #             mask = np.zeros(nTrials, dtype=bool)
    #             # Find the indices of the trlist that correspond to the condition
    #             indy = (trialsummary['port'] == lr) & (trialsummary['cond'] == iCond) & (trialsummary['resp'] == resp)
    #             mask[indy] = True
    #             # Continue onto next condition if no trials exist
    #             if sum(mask) == 0:
    #                 continue
    #
    #             # Create a new list based of only that condition
    #             cond_MAPs = [trMAPs[i] for i in range(nTrials) if mask[i]]
    #             cond_MAPs.sort(key=len,reverse=True)
    #             max_tsteps = len(cond_MAPs[0])
    #
    #             # Get a list of trial lengths
    #             cond_ts = [len(trMAPs[i]) for i in range(nTrials) if mask[i]]
    #             cond_ts.sort(key=int,reverse=True)
    #             xmax = np.minimum(max_tsteps, int(50*round(1.5*np.median(cond_ts)/50)) )
    #
    #             # Create a numpy array with padded NaNs so we can look at all
    #             # of the trials of a particular condition in a heatmap
    #             MAP_padded = np.empty((sum(mask),max_tsteps))
    #             MAP_padded[:] = np.nan
    #             for ii, MAP in enumerate(cond_MAPs):
    #                 MAP_padded[ii][:len(MAP)] = MAP
    #
    #             # Plot the discrete state sequence of the collection of trials
    #             title = '{} Condition, MAP estimates for {} {} trials of the {} Active Odor Port for {}'.format(Cond_Dict[iCond],sum(mask),Resp_Dict[resp],Port_Dict[lr],mouseID)
    #             #  title = 'MAP estimates for {} {}, {} trials of the {} Active Odor Port'.format(sum(mask),Cond_Dict[iCond],Resp_Dict[Port_Dict[lr])
    #
    #             plot_z_samples(MAP_padded,used_states,xmax,title = title,pdf = pdfdoc)
    #             # pdb.set_trace()
    #
    # if SAVEFIG:
    #     # Close PDF file
    #     pdfdoc.close()

#-------------------------------------------------------------------------------
def plot_state_seq_posteriors(trPosteriors,used_states,trialsummary,args,mouseID,Plot_Dir=None,fname=None, SAVEFIG=False):
    # Get number of trials from trial summary data frame
    nTrials = len(trialsummary)
    timestr = time.strftime('%Y%m%d_%H%M')
    if SAVEFIG:
        if Plot_Dir is None:
            Plot_Dir = './plots'

        # Create a pdf document for plots
        if fname is None:
            # Create a pdf document for plots
            if args['model'] is 'AR-HDP-HMM':
                fname = 'MAP_StateSeq_{}_K{:.0f}G{:.0f}_{}.pdf'.format(mouseID,args['K'],args['G'],timestr)
            else:
                fname = 'MAP_StateSeq_{}_N{}_{}.pdf'.format(mouseID,args['Nmax'],timestr)
        pdfdoc = PdfPages(os.path.join(Plot_Dir,fname))
    else:
        pdfdoc = None

    #Apply mask to MAP state sequence, which is a copy with local scope
    for MAP,mask in zip(trMAPs,trMASKs):
        MAP[mask] = np.nan

    #Plot all MAP sequences on one plot
    trMAPs_sorted = sorted(trMAPs,key=len,reverse=True)
    max_tsteps = len(trMAPs_sorted[0])

    MAP_padded = np.empty((nTrials,max_tsteps))
    MAP_padded[:] = np.nan
    for ii, MAP in enumerate(trMAPs_sorted):
        MAP_padded[ii][:len(MAP)] = MAP

    # Get a list of trial lengths
    MAP_ts = [len(trMAPs[i]) for i in range(nTrials)]
    MAP_ts.sort(key=int,reverse=True)
    xmax = int(50*round(1.5*np.median(MAP_ts)/50))
    if xmax > max_tsteps:
        xmax = max_tsteps

    # Plot the discrete state sequence of the collection of trials
    title = 'MAP estimates for all trials of {}'.format(mouseID)
    plot_z_samples(MAP_padded,used_states,xmax,title = title,pdf = pdfdoc)

    # Plot ARHMM MAP State Sequences
    for resp in Resp_Dict:
         # Loop over conditions
        for iCond in Cond_Dict:
    #         # Loop over active odor porth
            for lr in Port_Dict:
                mask = np.zeros(nTrials, dtype=bool)
                # Find the indices of the trlist that correspond to the condition
                indy = np.where((trialsummary['port'] == lr) & (trialsummary['cond'] == iCond) & (trialsummary['resp'] == resp))
                mask[indy] = True
                # Continue onto next condition if no trials exist
                if sum(mask) == 0:
                    continue

                # Create a new list based of only that condition
                cond_MAPs = [trMAPs[i] for i in range(nTrials) if mask[i]]
                # Sort based on length
                cond_MAPs.sort(key=len,reverse=True)
                max_tsteps = len(cond_MAPs[0])

                # Get a list of trial lengths
                cond_ts = [len(trMAPs[i]) for i in range(nTrials) if mask[i]]
                cond_ts.sort(key=int,reverse=True)
                xmax = int(50*round(1.5*np.median(cond_ts)/50))
                if xmax > max_tsteps:
                    xmax = max_tsteps

                # Create a numpy array with padded NaNs so we can look at all
                # of the trials of a particular condition in a heatmap
                MAP_padded = np.empty((sum(mask),max_tsteps))
                MAP_padded[:] = np.nan
                for ii, MAP in enumerate(cond_MAPs):
                    MAP_padded[ii][:len(MAP)] = MAP

                # Plot the discrete state sequence of the collection of trials
                title = '{} Condition, MAP estimates for {} {} trials of the {} Active Odor Port for {}'.format(Cond_Dict[iCond],sum(mask),Resp_Dict[resp],Port_Dict[lr],mouseID)
                #  title = 'MAP estimates for {} {}, {} trials of the {} Active Odor Port'.format(sum(mask),Cond_Dict[iCond],Resp_Dict[Port_Dict[lr])

                plot_z_samples(MAP_padded,used_states,xmax,title = title,pdf = pdfdoc)
                # pdb.set_trace()

    if SAVEFIG:
        # Close PDF file
        pdfdoc.close()


#-------------------------------------------------------------------------------
def plot_trans_matrix(trans_matrix, state_usages, log_TM=True, errorbars=False, Ws=None,
                      empirical=True, title='', Plot_Dir=None, fname=None, SAVEFIG=False):
    if SAVEFIG:
        if Plot_Dir is None:
            Plot_Dir = './plots'

        pdf = PdfPages(os.path.join(Plot_Dir,fname))
    else:
        pdf = None
    
    if log_TM:
        trans_matrix = np.log(trans_matrix)

    # Convert numpy arrays into Panda DataFrames
    tm = pd.DataFrame(trans_matrix)
    su = pd.DataFrame(state_usages)

    # Plotting Properties
    fig = plt.figure(figsize=(16,8))
    gs = gridspec.GridSpec(1,4,wspace=.5)
    fp = FontProperties()
    fp.set_weight("bold")
    if errorbars: #len(state_usages)>dis_k:
        # Calculate error bars for plot
        ci = 'sd'
    else:
        # Only 1 array of state usages
        ci = None

    # Draw a heatmap with the numeric values in each cell
    cmap = sns.cubehelix_palette(light=1, as_cmap=True)
    ax1 = fig.add_subplot(gs[0,0:2])
    if log_TM:
        ax1.set_title('log(Transition Probability Matrix)')
        lstr = 'log '
        sns.heatmap(tm, cmap=sns.color_palette("RdBu_r", 100),annot=False, fmt=".2f", linewidths=.5, ax=ax1,square=True,cbar_kws={'label': lstr + 'Probability','shrink': 0.68})
    else:
        ax1.set_title('Transition Probability Matrix')
        lstr = ''        
        sns.heatmap(tm, cmap=cmap,annot=False, fmt=".2f",vmin=0, vmax=1, linewidths=.5, ax=ax1,square=True,cbar_kws={'label': lstr + 'Probability','shrink': 0.68})

    # Plot overall state_usages
    pcolors = sns.xkcd_palette(color_names)
    ax2 = fig.add_subplot(gs[0,2:])
    ax2.set_xticks(np.arange(len(state_usages)))
    ax2.set_xticklabels(np.arange(len(state_usages)))
    if empirical:
        ax2 = sns.barplot(data =su,ci = ci,orient='h',palette=pcolors)
        ax2.set_title('State Usage')                
        ax2.set_xlabel('Probability')
        ax2.set_ylabel('State')
    else:
        ax2.bar(range(len(state_usages)), state_usages, color=pcolors)
#         ax2 = sns.barplot(x = np.arange(len(state_usages)),y = state_usages ,orient='h',palette=pcolors)
        ax2.set_title('Overall State Usage')        
        ax2.set_ylabel('Probability')
        ax2.set_xlabel('State')
#     ax2.axis('equal')

    # Set Super Title
    if title is not None:
        fig.suptitle(title)
    else:
        fig.suptitle('Overall ARHMM Fit')

    if pdf is not None:
        #pdf.savefig(fig)
        fpath = os.path.join(Plot_Dir,fname)
        fig.savefig(fpath)        

        #Close figures
        plt.close('all')
    else:
        fig.suptitle(fname)
        plt.show()

#-------------------------------------------------------------------------------
def construct_trans_matrices(arhmm,trMAPs,trans_matrix_mean,trialsummary,args,Plot_Dir=None):

    nTrials = len(trialsummary)
    used_states = sorted(arhmm.used_states)
    dis_k = len(used_states)

    ##======= Construct transition matrices & State Usages ======##
    # 3-3: Using the MAP-seq's in different conditions, calculate conditional
    # state-usages (% time-steps in each state) in each different condition.
    # 3-4: Using MAP-seqs, calculate the transition matrix for each condition
    timestr = time.strftime('%Y%m%d_%H%M')
    if Plot_Dir is None:
        Plot_Dir = './plots'

    if args['model'] is 'AR-HDP-HMM':
        fname = 'TransitionMatrices_A{:.0f}_K{:.0f}G{:.0f}_{}.pdf'.format(args['A'],args['K'],args['G'],timestr)
    else:
        fname = 'TransitionMatrices_{}_N{}_{}.pdf'.format(args['mouseID'],args['Nmax'],timestr)
    pdfdoc = PdfPages(os.path.join(Plot_Dir,fname))

    # Plot the mean transition matrix calculated from the Gibbs samples
    plot_trans_matrix(trans_matrix_mean, [arhmm.state_usages], dis_k,title='Transition Matrix calculated from Gibbs samples',pdf = pdfdoc)
    # Plot the transition matrix & state usage for the overall ARHMM fit
    plot_trans_matrix(arhmm.trans_distn.trans_matrix, [arhmm.state_usages], dis_k,pdf = pdfdoc)
    pdfdoc.close()

    trans_matrices = [[],[]]
    # Loop over responses
    for resp in Resp_Dict:
        # Loop over active odor port
        for lr in Port_Dict:
            # Calculate transition matrix per condition per response
            cond_trans_matrix = np.zeros((len(Cond_Dict),dis_k,dis_k))
            # Loop over conditions
            for iCond in Cond_Dict:
                # Reset Mask to False
                mask = np.zeros(nTrials, dtype=bool)

                # Find the indices of the trialsummary that correspond to the condition
                indy = np.where((trialsummary['cond'] == iCond) & (trialsummary['port'] == lr) & (trialsummary['resp'] == resp))
                mask[indy] = True
                # Continue onto next condition if no trials exist
                if sum(mask) == 0:
                    continue

                # Create a new list based on only that condition
                cond_MAPs = [trMAPs[i].copy() for i in range(nTrials) if mask[i]]

                # Calculate condition state usages
                cond_state_usages = np.array([[sum(MAP == s)/len(MAP) for s in used_states] for MAP in cond_MAPs])

                # Loop through the trials of this condition/response type
                for iTrial, MAP in enumerate(cond_MAPs):
                    for t in range(len(MAP)-1):
                        # Get the state at time t & t+1
                        s1,s2 = MAP[t],MAP[t+1]
                        # Get the indices associated with used_states
                        i1 = np.where(used_states == s1)
                        i2 = np.where(used_states == s2)
                        cond_trans_matrix[iCond,i1,i2] += 1

                # Divide each row by the number of transitions from that state
                tot_trans = np.sum(cond_trans_matrix[iCond],axis = 1)
                for i,rowsum in enumerate(tot_trans):
                    if rowsum == 0:
                        cond_trans_matrix[iCond,i,:] = 0
                    else:
                        cond_trans_matrix[iCond,i,:] = cond_trans_matrix[iCond,i,:]/rowsum

                title = '{} Condition, {} Active Odor Port, {} Response'.format(Cond_Dict[iCond],Port_Dict[lr],Resp_Dict[resp])
                # Plot transition matrix and state usage for this condition
                plot_trans_matrix(cond_trans_matrix[iCond], cond_state_usages,dis_k, title = title,pdf = pdfdoc)
                # End of Cond_Dict loop
            trans_matrices[resp].append(cond_trans_matrix)
            # End of Pord_Dict loop
        # End of Resp_Dict loop
    pdfdoc.close()

    return trans_matrices

#-------------------------------------------------------------------------------
def plot_model_convergence(stateseq_smpls, AB_smpls, trans_matrix_smpls, GibbsLLs, 
                           used_states, Plot_Dir, fname, SAVEFIG=True):
    
    if SAVEFIG:
        # Create a pdf document for plots
        timestr = time.strftime('%Y%m%d_%H%M')
        
        # Create a pdf document for plots
        pdfdoc = PdfPages(os.path.join(Plot_Dir,fname))

    # Get some shapes
    nTrials = len(stateseq_smpls)
    nGibbs, K, D_obs, _ = AB_smpls.shape
    d2 = D_obs**2
    num_states = K
    n2 = num_states**2
    fdim = D_obs*(D_obs+1)
    
    #Plot llhood of gibb samples
    fig1 = plt.figure(figsize = (10,8))
#     fig1.subplots_adjust(hspace=.2,wspace=.2)#.3)
#     ax = fig1.add_subplot(121)
    plt.plot(np.arange(nGibbs), GibbsLLs,'ok-')
    plt.xlabel('Gibbs Sample Iteration')
    plt.ylabel('Log-Likelihood')
    plt.title('Model Convergence ')
    if SAVEFIG:
        # Save figures
        pdfdoc.savefig(fig1)
    #Past way of displaying MAPsequences in sorted order
    cc, colors_shf = get_colors(len(used_states))

#     sfig = plt.figure(figsize=figsize)
#     sfig.subplots_adjust(hspace=.4,wspace=.4)#.3)
#     sfigb = plt.figure(figsize=figsize)
#     sfigb.subplots_adjust(hspace=.4,wspace=.4)#.3)

    IDxD = np.hstack((np.eye(D_obs),np.zeros((D_obs,1)))).flatten('F')

    # Loop over the different states
    for sp,state in enumerate(used_states):
        ABi = AB_smpls[:,state]
#         ax1 = sfig.add_subplot(*rowcol,sp+1) #plt.subplot(rows,cols,sp+1)

        AB_raster = np.empty((fdim,nGibbs))
        # Loop over Gibbs samples of each state matrix A and flatten
        for ii,sampleAB in enumerate(ABi):
            AB_raster[:,ii] = sampleAB.flatten('F').copy()

        sfig,(ax1,ax2) = plt.subplots(1,2,figsize=(16,6),gridspec_kw = {'wspace': 0.2})
        for iRow in range(d2): #range(fdim):
            ax1.plot(AB_raster[iRow,:]-IDxD[iRow])
        ax1.set_xlabel('Gibbs Sample Iteration')
        ax1.set_ylabel('Model Parameter Value')
        ax1.set_title('Elements of A\'s')

#         ax2 = sfigb.add_subplot(*rowcol,sp+1) #plt.subplot(rows,cols,sp+1)
        for iRow in range(d2,fdim):
            ax2.plot(AB_raster[iRow,:])
        ax2.set_xlabel('Gibbs Sample Iteration')
        ax2.set_ylabel('Model Parameter Value')
        ax2.set_title('Elements of b\'s')
        sfig.suptitle('Parameter Convergence for state {} AR Parameters'.format(state))
        if SAVEFIG:
            pdfdoc.savefig(sfig)
            plt.close(sfig)
        
#     sfig.suptitle('Parameter Convergence for Elements of A\'s')
#     sfigb.suptitle('Parameter Convergence for Elements of b\'s')

    # Plot how the transition matrix changes over Gibbs samples
    INxN = np.eye(num_states).reshape((-1,1))#flatten('F')
    T_raster = np.empty((n2,nGibbs))
    for ii in range(nGibbs):
        sampleT = trans_matrix_smpls[ii]
        T_raster[:,ii] = sampleT.flatten('F').copy()

    T_raster = T_raster -INxN
    max_y = T_raster[:,10:].max()
    min_y = T_raster[:,10:].min()
    
    fig4 = plt.figure(figsize=(10,5))
    ax = fig4.add_subplot(111)
    for iRow in range(n2):
        ax.plot(T_raster[iRow,:])
    ax.set_ylim([min_y,max_y])
    ax.set_xlabel('Gibbs Sample Iteration')
    ax.set_ylabel('Transition Matrix Element Value')
    ax.set_title('Transition Matrix convergence')

    if SAVEFIG:
        # Save figures
#         pdfdoc.savefig(fig1)
#         pdfdoc.savefig(sfig)
#         pdfdoc.savefig(sfigb)
        pdfdoc.savefig(fig4)
               
        # Close PDF file
        pdfdoc.close()
        
        # Close figure
        plt.close('all')

#-------------------------------------------------------------------------------
def plot_arhmm_parameters(ABs, args, samp_rate=80, Plot_Dir=None, fname=None,
                          SAVEFIG=False, x_units="pixels"): # SSM=False):

#    if SSM: #input [As,bs] in ssm library's notation for ABs
#        ABs = [np.hstack((A,b)) for A, b in zip(ABs[0],ABs[1])]

    if SAVEFIG:
        # Create a pdf document for plots
        timestr = time.strftime('%Y%m%d_%H%M')
        if Plot_Dir is None:
            Plot_Dir = './plots'

        # Create a pdf document for plots
        if fname is None:
            if args['model'] is 'AR-HDP-HMM':
                fname = 'arhmm_Parameters_A{:.0f}_K{:.0f}G{:.0f}_{}.pdf'.format(args['A'],args['K'],args['G'],timestr)
            else:
                fname = 'arhmm_Parameters_N{}_{}.pdf'.format(args['Nmax'],timestr)

        pdfdoc = PdfPages(os.path.join(Plot_Dir,fname))

    # plot in units of arena length regardless of x_units of args
    if x_units == "pixels":
        norm_factor = 1
    elif x_units == "arena length":
        norm_factor = Arena_Length
    else:
        raise NotImplementedError
  
    # Plotting Properties
    fp = FontProperties()
    fp.set_weight("bold")
    # cmap = sns.cubehelix_palette(light=1, as_cmap=True)
    cmap = sns.color_palette("RdBu_r")
    xstar = []
    for state,ab in enumerate(ABs):
        # Convert numpy arrays into Panda DataFrames & separate A & B matrices
        A = pd.DataFrame(ab[:,:-1])
        B = pd.DataFrame(ab[:,-1])
        D_obs = A.shape[0]

        # Calculate Eigenvalues of A
        w,v = np.linalg.eig(A)
        xstar = np.matmul(np.linalg.inv(np.eye(D_obs)-A),B)

        #Create new figure
        fig = plt.figure(figsize=(10,5))
        gs = gridspec.GridSpec(1,5)
        fig.suptitle('ARHMM Parameters for state {}'.format(state))
        # Plot Eigenvalues of A
        ax1 = fig.add_subplot(gs[0,0])
        ax1.plot(np.real(w),np.imag(w),'.k',MarkerSize=9)
        axlims = ax1.axis()
        ax1.add_patch(plt.Circle((0, 0), radius=1, edgecolor='b', facecolor='None',LineWidth=2))
        # ax1.set_aspect('equal', 'box')
        ax1.axis(axlims)        
        # ax1.axis('equal')
        ax1.set_xlabel('Real')
        ax1.set_ylabel('Imag')
        ax1.set_title('Eigenvalues of A')
        ax1.grid()

        # Plot ARHMM matrix A
        ax2 = fig.add_subplot(gs[0,1:4])
        sns.heatmap((A-np.eye(D_obs))*samp_rate, cmap=cmap,annot=True, fmt=".2f", linewidths=.5, ax=ax2,square=True,cbar=False)
        ax2.set_title('(A-I)/dt Matrix')

        ax3 = fig.add_subplot(gs[0,4])
        sns.heatmap(xstar/norm_factor, annot=True, fmt=".2f", linewidths=.5, ax=ax3,cbar=False,square=True)
        ax3.set_title('x*')

        if SAVEFIG:
            pdfdoc.savefig(fig)
            plt.close(fig)
        else:
            plt.show()
            
    if SAVEFIG:
        # Close PDF file
        pdfdoc.close()

#-------------------------------------------------------------------------------
def plot_arena(data=None,xstar=None,used_states = None,ax=None,fig=None,linecolor='k'):
    fig = fig if fig else plt.figure(figsize=(10,5), norm_factor=1)
    ax = ax if ax else fig.add_subplot(111)
    ax.set_axis_off()
    if data is not None:
        for i in range(0,data.shape[1],2):
            ci = int((i%data.shape[1])/2)
            plt.scatter(data[:,i],data[:,i+1],c=colors[ci],s=10)
        # plt.scatter(data[:,0],data[:,1],c=colors.colors[0],s=10)
        # plt.scatter(data[:,2],data[:,3],c=colors.colors[1],s=10)
        # plt.scatter(data[:,4],data[:,5],c=colors.colors[2],s=10)

    # Plot the boundaries of the arena
    x_lims = np.array([80, 1104])/norm_factor
    y_lims = np.array([44, 663])/norm_factor
    x_decision = 421/nomr_factor
    init_port = np.array([1127,349])/norm_factor
    right_port = np.array([1127,349])/norm_factor    
    left_port = np.array([173,24])/norm_factor  
    pl_x_lims = np.array([-620,1820])/norm_factor
    pl_y_lims = np.array([-370,1070])/norm_factor
    
    plt.plot(x_lims,[y_lims[0],y_lims[0]],'-'+linecolor,LineWidth = 3,zorder=1)
    plt.plot(x_lims,[y_lims[1],y_lims[1]],'-'+linecolor,LineWidth = 3,zorder=1)
    plt.plot([x_lims[0],x_lims[0]], ylims,'-'+linecolor,LineWidth = 3,zorder=1)    
    plt.plot([x_lims[1],x_lims[1]], ylims,'-'+linecolor,LineWidth = 3,zorder=1)        
    plt.plot([x_decision,x_decision],y_lims,'--'+linecolor,LineWidth = 3,zorder=1)
    plt.plot([x_lims[0],x_decision],[y_lims.mean(),y_lims.mean()],'--'+linecolor,LineWidth = 3,zorder=1)
    plt.plot(*init_port, 'o'+linecolor,MarkerSize = 10)#initiation port
    plt.plot(*right_port, 's'+linecolor,MarkerSize = 10)#right reward port
    plt.plot(*left_port, 's'+linecolor,MarkerSize = 10)#left reward port
    # # plt.plot(789,354,'ko') #approximate average body-coordinates
    # # plt.plot(789+77,354,'kx') # + average head-body distance along x
    # # plt.plot(789+77+71,354,'k+') # + average nose-head distance along x
    # plt.plot(762,350,'ko') #approximate average head-coordinates
    # plt.plot(762-77,354,'kx') # - average head-body distance along x
    # plt.plot(762+71,354,'k+') # + average nose-head distance along x
    plt.xlim(pl_x_lims)
    plt.ylim(pl_y_lims)
    plt.xticks([])
    plt.yticks([])
    plt.axis('equal')

    if xstar is not None:
        xl = np.array([-600,1800])/norm_factor
        yl = np.array([-350,1050])/norm_factor
        ac = np.array([592,354])/norm_factor
        for state, x in zip(used_states,xstar):
            # If its within a certain bound of the arena, plot the fixed point
            if (xl[0] <= x[2] <= xl[1]) and (yl[0] <= x[3] <= yl[1]):
                plt.scatter(x[2],x[3],c=colors[state],edgecolors=linecolor,s=100,marker=(5,1),zorder=2)
            else:
            # If not, plot it at a boundary
                m = (x[3]-ac[1])/(x[2]-ac[0])
                b = x[3]-m*x[2]

                #If x value is within bounds & the y-value is above/below plot
                if (xl[0] <= x[2] <= xl[1]):
                    plt.scatter(x[2],m*x[2]+b,c=colors[state],edgecolors=linecolor,s=100,marker=(5,1),zorder=2)
                #If the x-value is right of plot
                elif (x[2] > xl[1]):
                    y=m*xl[1]+b
                    if y > yl[1]:
                        y=yl[1]
                    elif y < yl[0]:
                        y=yl[0]

                    plt.scatter(xl[1],y,c=colors[state],edgecolors=linecolor,s=100,marker=(5,1),zorder=2)
                #If the x-value is left of plot
                elif (x[2] < xl[0]):
                    y=m*xl[1]+b
                    if y > yl[1]:
                        y=yl[1]
                    elif y < yl[0]:
                        y=yl[0]
                    plt.scatter(xl[0],y,c=colors[state],edgecolors=linecolor,s=100,marker=(5,1),zorder=2)

        plt.xlim(pl_x_lims)
        plt.ylim(pl_y_lims)
        patches = [mpatches.Patch(color=colors[s], label="State {:.0f} : ({:.0f},{:.0f}) ".format(s,x[2],x[3])) for s,x in zip(used_states,xstar)]
        # put those patched as legend-handles into the legend
        plt.legend(handles=patches, loc=4)
        ax.set_title('ARHMM Fixed Points')

#-------------------------------------------------------------------------------
def plot_fixed_points(As_mean,xstar,used_states,used_states_sorted,args=None,
                      NoseBody=False, Plot_Dir=None, fname=None, SAVEFIG=False, edgecolor='k',
                      ORDER_STATES=False, x_units="pixels"):

    #Get Plot_Dir to save plots
    timestr = time.strftime('%Y%m%d_%H%M')
    if Plot_Dir is None:
        Plot_Dir = './plots'

    if SAVEFIG:
         # Create a pdf document for plots
        if fname is None:
            if args['model'] is 'AR-HDP-HMM':
                fname = 'FixedPoints_K{:.0f}G{:.0f}_{}.pdf'.format(args['K'],args['G'],timestr)
            else:
                fname = 'FixedPoints_N{}_{}.pdf'.format(args['Nmax'],timestr)
        pdfdoc = PdfPages(os.path.join(Plot_Dir,fname))

    if x_units == "pixels":
        norm_factor = 1
    elif x_units == "arena length":
        norm_factor = Arena_Length
    else:
        raise NotImplementedError
        
    #cc, colors_shf = shuffle_colors(used_states_sorted)
    cc, colors_shf = get_colors(len(used_states_sorted))

    #Plot fixed points
    sfig = plt.figure(figsize=(10,5))
    ax = sfig.add_subplot(111)
    #Plot Arena
    plot_arena(ax=ax,fig=sfig, linecolor=edgecolor, norm_factor=norm_factor)

    #Loop over autoregressive matrices
    fpt_stability = []
    patches = []
    #for i, (AB,x,s) in enumerate(zip(As_mean,xstar,used_states)):
    for i, s in enumerate(used_states_sorted):
        AB, x = As_mean[s], xstar[s]
        A = AB[:,:-1]
        B = AB[:,-1]
        scol = i #s

        #Get eigenvalues of A
        ww,vv = np.linalg.eig(A)

        #Sort based on absolute value
        # sorted_eig = np.array(sorted(np.real(ww),key=abs,reverse=True))
        # fpt = (sorted_eig % 1) - sorted_eig
        # for stability in fpt:
        #     if stability == -1 or stability == 2:
        #         fpt_str += '-'
        #     else:
        #         fpt_str += '+'
        sorted_eig = np.array(sorted(np.abs(ww),reverse=True))
        fpt_str=''
        for stability in sorted_eig:
            if stability < 1:
                fpt_str += '-'
            else:
                fpt_str += '+'
        fpt_stability.append(fpt_str)

        # plt.plot([x[0],x[2]],[x[1],x[3]],'-k')
        # plt.plot([x[2],x[4]],[x[3],x[5]],'-k')
        if not NoseBody:
            plt.scatter(x[2],x[3],c=colors_shf[scol],s=40,marker='X', zorder=2)
            plot_title = 'AR Fixed Points (Head coordinates)'
        else:
            plt.plot(x[[0,2,4]],x[[1,3,5]],'-x',c=colors_shf[scol], zorder=2)
            plt.scatter(x[4],x[5],c=colors_shf[scol],s=40,marker='o', zorder=2)
            plot_title = 'AR Fixed Points'
        # plt.scatter(x[2],x[3],c=colors_shf[scol],s=40,marker='H', zorder=2)
        # plt.scatter(x[4],x[5],c=colors_shf[scol],s=40,marker='D', zorder=2)

        patches.append(mpatches.Patch(color=colors_shf[scol], label='State {:2d}: Fixed Pts: {}'.format(s,fpt_str)))
    #Make sure limits are correct
    plt.xlim(np.array([-620,1820])/norm_factor)
    plt.ylim(np.array([-370,1070])/norm_factor)

    # if not ORDER_STATES:
    #     patches = [mpatches.Patch(color=colors_shf[s], label='State {:2d}: Fixed Pts: {}'.format(s,fp)) for s,fp in zip(used_states,fpt_stability)]
    # else:
    #     patches = [mpatches.Patch(color=colors_shf[used_states_sorted[s]], label='State {:2d}: Fixed Pts: {}'.format(used_states_sorted[s],fp)) for s,fp in zip(used_states,fpt_stability)]
    # put those patched as legend-handles into the legend
    plt.legend(handles=patches, loc=4)
    ax.set_title(plot_title)

    #Save Fig
    if SAVEFIG:
        fpath = os.path.join(Plot_Dir,fname)
        sfig.savefig(fpath)
        plt.close('all')
    else:
        plt.show()

#-------------------------------------------------------------------------------
def plot_AR_action(As_mean, used_states_sorted, state_usages, timesteps=(4,1),
                   ORDER_STATES=False, AXIS_OFF=False,
                   figsize=(15,6), rowcol=(2,5), args=None, Plot_Dir=None,
                   fname=None, SAVEFIG=False, edgecolor='k', x_units="pixels"):

    #Get Plot_Dir to save plots
    if Plot_Dir is None:
        Plot_Dir = './plots'

    if SAVEFIG:
        if fname is None:
            timestr = time.strftime('%Y%m%d_%H%M')
             # Create a pdf document for plots
            if args['model'] is 'AR-HDP-HMM':
                fname = 'FixedPoints_K{:.0f}G{:.0f}_{}.pdf'.format(args['K'],args['G'],timestr)
            else:
                fname = 'FixedPoints_N{}_{}.pdf'.format(args['Nmax'],timestr)
        pdfdoc = PdfPages(os.path.join(Plot_Dir,fname))

    if x_units == "pixels":
        norm_factor = 1
    elif x_units == "arena length":
        norm_factor = Arena_Length
    else:
        raise NotImplementedError
        
    state_usages = np.array(np.round(100*state_usages),dtype=np.int32)

    # if not ORDER_STATES:
    #     cc, colors_shf = shuffle_colors(used_states_sorted)
    # else:
    #     cc, colors_shf = shuffle_colors(used_states)
    #     As_mean = [As_mean[used_states_sorted[i]] for i in range(len(used_states)) ]

    if isinstance(used_states_sorted,list):
        cc, colors_shf = get_colors(len(used_states_sorted))
        #cc, colors_shf = shuffle_colors(used_states_sorted)
    else:
        K = used_states_sorted
        cc, colors_shf = get_colors(K)
        used_states_sorted = list(range(K))
    

    if len(used_states_sorted)>rowcol[0]*rowcol[1]:
        rowcol = list(rowcol)
        rowcol[0] = 1 + len(used_states_sorted)/rowcol[1]

    step = timesteps[1]
    timesteps = timesteps[0]
    time_colors = np.linspace(0,1,timesteps+1).reshape((-1,1)) * np.ones((1,3))

    lNH = 71/norm_factor #average nose-head distance
    lBH = 77/norm_factor #average body-head distance
    lBN = lBH + lNH
    Bshift = 40/norm_factor
    xH, yH = 762/norm_factor, 350/norm_factor #approximate average head coordinates
    xB, yB = 789/norm_factor, 354/norm_factor #approximate average body coordinatesa
    x0s = np.array([[xB+Bshift,yB, xB+Bshift+lBH,yB, xB+Bshift+lBN,yB ,1],  #pointing towards initiation port
                    [xB-Bshift,yB, xB-Bshift-lBH,yB, xB-Bshift-lBN,yB ,1],  #pointing aways from initiation port
                    [xB,yB+Bshift, xB,yB+Bshift+lBH, xB,yB+Bshift+lBN ,1],  #pointing towards right wall
                    [xB,yB-Bshift, xB,yB-Bshift-lBH, xB,yB-Bshift-lBN ,1]]) #pointing towards left wall
    # #use this instead for Matt:
    #x0s = np.array([[xB-Bshift,yB, xB-Bshift-lBH,yB, xB-Bshift-lBN,yB ,1]])  #pointing aways from initiation port


    sfig = plt.figure(figsize=figsize)
    sfig.subplots_adjust(hspace=.4,wspace=.4)#.3)
    # #Plot Arena
    # plot_arena(ax=ax,fig=sfig, linecolor=edgecolor, norm_factor=norm_factor)
    def plotBHN(ax,x,t,s,ms=10,lw=2.5):
        xp = x.reshape((3,2))
        ax.plot(xp[:,0],xp[:,1],'-o',c=colors_shf[s],markersize=ms/2, zorder=2,linewidth=lw) #plot B-H-N
        ax.plot(xp[-1,0],xp[-1,1],'.',c=time_colors[t],markersize=ms, zorder=3) #plot time-colored Nose
        ax.plot(xp[-2,0],xp[-2,1],'.',c=time_colors[t],markersize=ms, zorder=3) #plot time-colored Head
        ax.plot(xp[-3,0],xp[-3,1],'.',c=time_colors[t],markersize=ms, zorder=3) #plot time-colored Body

    #Loop over MC states and their autoregressive matrices
    #for sp, (Ab,s) in enumerate(zip(As_mean,used_states)):
    for sp, s in enumerate(used_states_sorted):
        Ab = As_mean[s]

        ax = sfig.add_subplot(*rowcol,sp+1) #plt.subplot(rows,cols,sp+1)
        if AXIS_OFF: ax.set_axis_off()
        for x0 in x0s: #four different initial mouse poses to act on by Ab
            x = x0
            plotBHN(ax,x[:-1],0,sp)#,s)
            for t in range(timesteps):
                x = Ab @ x
                x = np.concatenate((x,[1]))
                #pdb.set_trace()
                if (t+1) % step ==0: #only plot every "step" steps
                    plotBHN(ax,x[:-1],t+1,sp)#,s)

        ax.axis('equal')
        ax.axis('off')        
        ax.grid(False)
        # ax.set_xlim([450/norm_factor,800/norm_factor])
        # ax.set_ylim([200/norm_factor,550/norm_factor])
        # ax.set_xticks([500/norm_factor,600/norm_factor,700/norm_factor])
        # ax.set_yticks([300/norm_factor,400/norm_factor,500/norm_factor])
        # #ax.set_facecolor((1.,1.,1.))

        # s1 = s
        # if ORDER_STATES: s1 = used_states_sorted[s]
        # ax.set_title('state {}'.format(s1),color=colors_shf[s])

        ax.set_title('state {}, usage = {} %'.format(s,state_usages[s]),color=colors_shf[sp])

        # #Make sure limits are correct
        # plt.xlim([-620/norm_factor,1820/norm_factor])
        # plt.ylim([-370/norm_factor,1070/norm_factor])

    #Save Fig
    if SAVEFIG:
        fpath = os.path.join(Plot_Dir,fname)
        sfig.savefig(fpath)
        plt.close('all')
    else:
        sfig.suptitle(fname)
        plt.show()
        return (sfig,ax)

#-------------------------------------------------------------------------------
def plot_AR_noise(Sigmas, nus=None, x_units="pixels", samp_rate=80, Plot_Dir=None, fname=None, SAVEFIG=False):
    if SAVEFIG:
        if Plot_Dir is None:
            Plot_Dir = './plots'

        pdf = PdfPages(os.path.join(Plot_Dir,fname))
    else:
        pdf = None

    extra = 1 if nus is not None else 0

    K = Sigmas.shape[0]
    D_obs = Sigmas.shape[1]
    if D_obs != 6:
        raise NotImplementedError('Currently only implemented for D_obs = 6 (Body-Head-Nose cartesian coordinates).')
        
    # plot in units of arena length regardless of x_units of args
    ystr = '(units of arena length)' 
    if x_units == "pixels":
        norm_factor = Arena_Length / np.sqrt(samp_rate)
    elif x_units == "arena length":
        norm_factor = 1 / np.sqrt(samp_rate)
    else:
        raise NotImplementedError
        
    inds = np.arange(D_obs)
    Vars = Sigmas[:,inds,inds] #take the diagonal part of Sigma matrices for all HMM states
    Vars_B = Vars[:,:2]
    Vars_H = Vars[:,2:4]
    Vars_N = Vars[:,4:]    
    rms_B = np.sqrt(Vars_B.sum(axis=1)) / norm_factor
    rms_H = np.sqrt(Vars_H.sum(axis=1)) / norm_factor
    rms_N = np.sqrt(Vars_N.sum(axis=1)) / norm_factor
    
    fig, axs = plt.subplots(3+extra, 1, figsize=(6,20)) # fig, ax =plt.subplots(figsize=(6, 4))
    axs[0].bar(range(K), rms_B, color=colors[:K])
    axs[0].set(xlabel='HMM states')
    axs[0].set(ylabel='Body AR Noise (rms)'+ystr)    
    axs[1].bar(range(K), rms_H, color=colors[:K])
    axs[1].set(xlabel='HMM states')
    axs[1].set(ylabel='Head AR Noise (rms)'+ystr)    
    axs[2].bar(range(K), rms_N, color=colors[:K])
    axs[2].set(xlabel='HMM states')
    axs[2].set(ylabel='Nose AR Noise (rms)'+ystr)    
    if nus is not None:
        axs[3].bar(range(K), nus, color=colors[:K])
        axs[3].set(xlabel='HMM states')
        axs[3].set(ylabel='t-distribution nu\'s')    
        
    if pdf is not None:
        #pdf.savefig(fig)
        fpath = os.path.join(Plot_Dir,fname)
        fig.savefig(fpath)        

        #Close figures
        plt.close('all')  
    else:
        fig.suptitle(fname)
        plt.show()
    
#-------------------------------------------------------------------------------
def plot_angle_hist(trMAPs,angle_list,lr_list,used_states,used_states_sorted,args,mouseID,Plot_Dir=None):
    timestr = time.strftime('%Y%m%d_%H%M')
    if Plot_Dir is None:
        Plot_Dir = './plots'

    cc, colors_shf = shuffle_colors(used_states_sorted)

    # Create a pdf document for plots
    if args['model'] is 'AR-HDP-HMM':
        fname = 'AngleHistograms_K{:.0f}G{:.0f}_{}.pdf'.format(args['K'],args['G'],timestr)
    else:
        fname = 'AngleHistograms_{}_N{}_{}.pdf'.format(mouseID,args['Nmax'],timestr)
    pdfdoc = PdfPages(os.path.join(Plot_Dir,fname))

    #Create lusts to hold angle information
    l_theta = [[] for s in used_states]
    r_theta = [[] for s in used_states]
    l_phi = [[] for s in used_states]
    r_phi = [[] for s in used_states]
    l_ba = [[] for s in used_states]
    r_ba = [[] for s in used_states]

    #Loop over data and calculate state statistics
    for MAPseq,angles,lr in zip(trMAPs,angle_list,lr_list):
        #Separate angles into states
        for i,s in enumerate(used_states):
            indy = np.where(MAPseq == s)[0]
            #Separate between left/right active odor ports
            if lr == 2:
                l_phi[i].extend(angles[indy,0]*180/np.pi)
                l_theta[i].extend(angles[indy,1]*180/np.pi)
                l_ba[i].extend((angles[indy,0]-angles[indy,1]+np.pi)*180/np.pi)
            else:
                r_phi[i].extend(angles[indy,0]*180/np.pi)
                r_theta[i].extend(angles[indy,1]*180/np.pi)
                r_ba[i].extend((angles[indy,0]-angles[indy,1]+np.pi)*180/np.pi)

    #Plot Angle Usage per state and active odor port
    #Head-Body Angle
    pfig = plt.figure(figsize=(10,5))
    ps = gridspec.GridSpec(2,len(used_states))
    pfig.suptitle('Body-Head Angle Phi {}'.format(mouseID))

    for i,s in enumerate(used_states):
        axl = pfig.add_subplot(ps[0,i])
        plt.hist(l_phi[i],bins=180,color=colors_shf[s])
        axl.set_title('State {} Phi'.format(s))

        axr = pfig.add_subplot(ps[1,i])
        plt.hist(r_phi[i],bins=180,color=colors_shf[s])
        axr.set_title('State {} Phi'.format(s))

    #Save Fig
    pdfdoc.savefig(pfig)

    #Nose-Head Angle
    tfig = plt.figure(figsize=(10,5))
    ts = gridspec.GridSpec(2,len(used_states))
    tfig.suptitle('Nose-Head Angle Theta for {}'.format(mouseID))
    for i,s in enumerate(used_states):
        axl = tfig.add_subplot(ts[0,i])
        plt.hist(l_theta[i],bins=180,color=colors_shf[s])
        axl.set_title('State {} Theta'.format(s))

        axr = tfig.add_subplot(ts[1,i])
        plt.hist(r_theta[i],bins=180,color=colors_shf[s])
        axr.set_title('State {} Theta'.format(s))

    #Save Fig
    pdfdoc.savefig(tfig)

    #Body-Angle
    bfig = plt.figure(figsize=(10,5))
    bs = gridspec.GridSpec(2,len(used_states))
    bfig.suptitle('Body Angle for {}'.format(mouseID))
    for i,s in enumerate(used_states):
        axl = bfig.add_subplot(bs[0,i])
        plt.hist(l_ba[i],bins=180,color=colors_shf[s])
        axl.set_title('State {}'.format(s))

        axr = bfig.add_subplot(bs[1,i])
        plt.hist(r_ba[i],bins=180,color=colors_shf[s])
        axr.set_title('State {}'.format(s))

    #Save Fig
    pdfdoc.savefig(bfig)

    #Body-Angle
    sfig = plt.figure(figsize=(10,5))
    gs = gridspec.GridSpec(1,2)
    sfig.suptitle('Body Angle for {}'.format(mouseID))
    axl = sfig.add_subplot(gs[0,0])
    axr = sfig.add_subplot(gs[0,1])
    for i,s in enumerate(used_states):

        axl.hist(l_ba[i],bins=180,color=colors_shf[s])
        axl.set_title('State {}'.format(s))

        axr.hist(r_ba[i],bins=180,color=colors_shf[s])
        axr.set_title('State {}'.format(s))

    patches = [mpatches.Patch(color=colors_shf[i], label="State {}".format(s)) for s in used_states]
    plt.legend(handles=patches, loc=1)
    #Save Fig
    pdfdoc.savefig(bfig)

    pdfdoc.close()
    plt.close('all')

#-------------------------------------------------------------------------------
def plot_state_durations(state_duration_list, used_states, used_states_sorted,
             trialsummary, mouseID, samp_rate=80, Plot_Dir=None, fname=None, SAVEFIG=False):
    if SAVEFIG:
        if Plot_Dir is None:
            Plot_Dir = './plots'

        pdfdoc = PdfPages(os.path.join(Plot_Dir,fname))
    else:
        pdfdoc = None

    
    if isinstance(used_states_sorted,list):
        cc, colors_shf = shuffle_colors(used_states_sorted)
    else:
        K = used_states_sorted
        cc, colors_shf = get_colors(K)
        used_states = list(range(K))
            
    nTrials = len(state_duration_list)
    
    #Group together all control trials, irrespective of response
    fig1 = plt.figure(figsize=(12,5))
    gs = gridspec.GridSpec(2,len(used_states))
    leftright = 'turn' #'port'
    indy_l = np.where((trialsummary['port'] == 2) & (trialsummary['cond'] != 3))[0]
    indy_r = np.where((trialsummary['port'] == 1) & (trialsummary['cond'] != 3))[0]
    if leftright == 'turn':
        titletext = lambda s1, s2 : 'State Durations for {} left and {} right initial turns, non-control Trials'.format(s1,s2)
    elif leftright == 'port':
        titletext = lambda s1, s2 : 'State Durations for {} left and {} right reward port, non-control Trials'.format(s1,s2)
    fig1.suptitle(titletext(len(indy_l),len(indy_r)))
                   
    # Loop over active odor ports
    for lr in Port_Dict:
        # Find the indices of the trlist that correspond to the condition
        indy = np.where((trialsummary[leftright] == lr) & (trialsummary['cond'] != 3))[0]
        
        for ii,state in enumerate(used_states):
            s_dur = []
            for iTrial in indy:
                s_dur.extend(state_duration_list[iTrial][ii])
            
            s_dur = np.asarray(s_dur) / samp_rate
            ax = fig1.add_subplot(gs[lr-1,ii])
            sns.distplot(s_dur,color=colors[ii],ax=ax)
            ax.axvline(s_dur.mean(), color=colors_shf[state], linewidth=2)
            ax.text(0.2, 0.5, 'n = {}'.format(len(s_dur)), transform=ax.transAxes,fontdict={'fontweight': 'normal', 'fontsize': 10})
            ax.text(0.2, 0.6, '\u03BC = {:.2f}'.format(s_dur.mean()), transform=ax.transAxes,fontdict={'fontweight': 'normal', 'fontsize': 10})
            if lr == 1:
                ax.set_title('State {}'.format(state))
            else:
                ax.set_xlabel('duration (sec)')
            #ax.set_xlim([0,.5]) #([0,25])
            if ii == 0:
                ax.set_ylabel('{} Port Usage'.format(Port_Dict[lr]))
            else:
                ax.set_yticks([])
    if SAVEFIG:
        pdfdoc.savefig(fig1)
    else:
        plt.show()    

    ## ========= Plot state durations for various conditions =============#
    #Loop over responses
    for resp in Resp_Dict:
        # Loop over conditions
        for iCond in Cond_Dict:
            #Make figure
            fig = plt.figure(figsize=(12,5))
            gs = gridspec.GridSpec(2,len(used_states))

            # Loop over active odor porth
            for lr in Port_Dict:
                # Find the indices of the trlist that correspond to the condition
                indy = np.where((trialsummary['port'] == lr) & (trialsummary['cond'] == iCond) & (trialsummary['resp'] == resp))[0]

                fig.suptitle('State Durations for {} {}, {} Trials'.format(len(indy),Resp_Dict[resp],Cond_Dict[iCond]))
                #s_dur = []
                for ii,state in enumerate(used_states):
                    s_dur = []
                    for iTrial in indy:
                        s_dur.extend(state_duration_list[iTrial][ii])

                    ax = fig.add_subplot(gs[lr-1,ii])
                    if len(s_dur) < 5:
                        continue
                    
                    s_dur = np.asarray(s_dur) / samp_rate
                    sns.distplot(s_dur,color=colors_shf[state],ax=ax)
                    ax.axvline(s_dur.mean(), color=colors_shf[state], linewidth=2)
                    ax.text(0.2, 0.5, 'n = {}'.format(len(s_dur)), transform=ax.transAxes,fontdict={'fontweight': 'normal', 'fontsize': 10})
                    ax.text(0.2, 0.6, '\u03BC = {:.2f}'.format(s_dur.mean()), transform=ax.transAxes,fontdict={'fontweight': 'normal', 'fontsize': 10})
                    if lr == 1:
                        ax.set_title('State {}'.format(state))
                    else:
                        ax.set_xlabel('duration (sec)')
                    #ax.set_xlim([0,.5]) #([0,25])
                    if ii == 0:
                        ax.set_ylabel('{} Port Usage'.format(Port_Dict[lr]))
                    else:
                        ax.set_yticks([])

            if SAVEFIG: 
                #Save Figure
                pdfdoc.savefig(fig)
                plt.close(fig)
            else:
                fig.suptitle(fname)
                plt.show() # plt.show(block=False)
    if SAVEFIG:
        pdfdoc.close()
        plt.close('all')
#-------------------------------------------------------------------------------
def plot_EM_convergence(EM_lps, max_EM_iters, Plot_Dir=None, fname=None, SAVEFIG=False):
    if SAVEFIG:
        if Plot_Dir is None:
            Plot_Dir = './plots'
        # Create a pdf document for plots
        if fname is None: 
            raise NotImplementedError
        pdfdoc = PdfPages(os.path.join(Plot_Dir,fname))

    
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(len(EM_lps)), EM_lps,'b-o')
    ax.set_xlabel('EM iterations')
    ax.set_ylabel('log-posterior')
    ax.set_title('max EM iterations = {}'.format(max_EM_iters))
    #Save Fig
    if SAVEFIG:
#        pdfdoc.savefig(fig)
#        fig.savefig(pdfdoc)
        fpath = os.path.join(Plot_Dir,fname)
        fig.savefig(fpath)
        plt.close('all')
    else:
        fig.suptitle(fname)
        plt.show()
        
#-------------------------------------------------------------------------------
def plot_xval_lls_vs_K(lls, Ks, title_str=None, Plot_Dir=None, fname=None, SAVEFIG=False):
    """ plot log-likelihood vs K. lls.shape must be (len(Ks), xval_folds)"""
    
    if Plot_Dir is None:
        Plot_Dir = './plots'
    # Create a pdf document for plots
    if fname is None: 
        raise NotImplementedError
    pdfdoc = PdfPages(os.path.join(Plot_Dir,fname))
    
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(Ks, lls[:,:-1], 'bo',label='Kfolds')
    ax.plot(Ks, lls[:,-1], 'ro',label='Full Model Fit')
    ax.plot(Ks, np.mean(lls[:,:-1], axis=1), 'k.-', markersize=8,label='Mean across kfolds')    
    ax.set_ylabel('log-likelihood per time-step')
    ax.set_xlabel('# HMM states (K)')
    ax.legend()
    if title_str is not None:
        ax.set_title(title_str)
    #Save Fig
    if SAVEFIG:
#        pdfdoc.savefig(fig)
#        fig.savefig(pdfdoc)
        fpath = os.path.join(Plot_Dir,fname)
        fig.savefig(fpath)
        plt.close('all')
    else:
        plt.show()
    
        
#-------------------------------------------------------------------------------
def trialsum_barplots(trsum,cond=None,figsize=(10,5)):
    if cond is None:
        inds = trsum['cond'] < 3
    else:
        inds = trsum['cond']==cond#.nonzero()

    trsum1 = trsum[inds] #trsum.iloc[inds]
    #weird: the &-operands have to be in parentheses, in following lines
    sc = np.sum((trsum1['port']==trsum1['turn']) & (trsum1['resp']==1))
    dc = np.sum((trsum1['port']!=trsum1['turn']) & (trsum1['resp']==1))
    si = np.sum((trsum1['port']==trsum1['turn']) & (trsum1['resp']==0))
    di = np.sum((trsum1['port']!=trsum1['turn']) & (trsum1['resp']==0))
    cases = np.array([sc,dc,si,di],dtype=np.float64)
    # cases *= 100./np.sum(cases)
    # cases[:2] *= 100./np.sum(cases[:2])
    # cases[2:] *= 100./np.sum(cases[2:])
    cases[2:] *= 10
    cases = np.array(np.round(cases),dtype=np.int32)
    ind = np.arange(4)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1,3,1)
    #fig, (ax,ax2,ax3) = plt.subplots(1,3)
    sc, dc, si, di = ax.bar(ind, cases)
    sc.set_facecolor('r')
    dc.set_facecolor('k')
    si.set_facecolor('r')
    di.set_facecolor('k')
    ax.set_xticks(ind)
    ax.set_xticklabels(['S-C', 'D-C', '10x S-I', '10x D-I']); #Same, Different, Correct, Incorrect
    #ax.set_ylim([0, 100])

    ax2 = fig.add_subplot(1,3,2)
    corr_turn = np.sum((trsum1['port']==trsum1['turn']) )
    incorr_turn = np.sum((trsum1['port']!=trsum1['turn']) )
    cases = np.array([corr_turn,incorr_turn],dtype=np.float64)
    cases = np.array(np.round(cases),dtype=np.int32)
    ind = np.arange(2)
    c_turn, i_turn = ax2.bar(ind, cases)
    c_turn.set_facecolor('r')
    i_turn.set_facecolor('k')
    ax2.set_xticks(ind)
    ax2.set_xticklabels(['corr turn', 'incorr turn']);

    ax3 = fig.add_subplot(1,3,3)
    corr = np.sum((trsum1['resp']==1))
    incorr = np.sum((trsum1['resp']==0))
    cases = np.array([corr,incorr],dtype=np.float64)
    cases = np.array(np.round(cases),dtype=np.int32)
    ind = np.arange(2)
    c_turn, i_turn = ax3.bar(ind, cases)
    c_turn.set_facecolor('r')
    i_turn.set_facecolor('k')
    ax3.set_xticks(ind)
    ax3.set_xticklabels(['correct', 'incorrect']);

    if cond is not None:
        fig.suptitle('condition = ' + Cond_Dict[cond])
    else:
        fig.suptitle('condition = all interleaved')

    plt.show()
    
def save_plot_parameters(SaveDir, fname_sffx, llhood_tuple,state_usage,
                        opt, params_dict, state_perm, model_convergence, RunTime):
    
    ##===== Save model parameters =====##
    if opt['save']:
        results_fname = '-'.join(('fit_parameters',fname_sffx))
        ioh5.save(os.path.join(SaveDir, results_fname+'.h5'), \
                        {'ll_heldout':llhood_tuple[0], 'll_heldout_perstep':llhood_tuple[1],
                        'll_training':llhood_tuple[2], 'll_training_perstep':llhood_tuple[3],
                        'state_usage':state_usage, 'arhmm_params' : params_dict, 
                        'hyperparams': opt, 'state_perm': state_perm, 
                        'model_convergence': model_convergence, 'RunTime': RunTime}) 

    ##===== plot em convergence =====##
#     if opt['model_type'] == 'ARHMM_SL':
#         plot_EM_convergence(model_convergence, opt['EM_iters'], Plot_Dir=SaveDir,fname='-'.join(('Model_convergence',fname_sffx))+'.pdf', SAVEFIG=opt['save'])
        
    # plot AR parameters
#     ABs = params_dict['observations']['ABs']
#     plot_arhmm_parameters(ABs, None, Plot_Dir=SaveDir, fname='-'.join(('AR_parameters',fname_sffx))+'.pdf',
#                                 SAVEFIG=opt['save'], x_units=opt['x_units'])
#     plot_AR_action(ABs, opt['K'], state_usage, timesteps=(4,1), Plot_Dir=SaveDir,
#                  fname='-'.join(('AR_starplots',fname_sffx))+'.pdf', SAVEFIG=opt['save'], x_units=opt['x_units'])
    
#     ##===== plot AR noise =====##
#     if 'sqrt_Sigmas' in params_dict['observations'].keys():
#         L = params_dict['observations']['sqrt_Sigmas'] #cholesky decomp of covariance matrix (up to nus)
#         Sigmas = np.matmul(L, np.transpose(L, (0,2,1)))  #np.matmul(L, np.swapaxes(L, -1, -2))
#     elif '_log_sigmasq' in params_dict['observations'].keys():
#         Sigmas = np.array([np.diag(np.exp(log_s)) for log_s in params_dict['observations']['_log_sigmasq']])
    
#     #Only relevant for Scott Linderman's SSM code
#     if opt['robust'] & (opt['model_type'] == 'ARHMM_SL'):
#         nus = params_dict['observations']['nus']
#         nus_fac = nus/(nus-2)
#         if (nus_fac<0).any():
#             print('\nWarning: some nu\'s are smaller than 2!\n')
#         Sigmas = Sigmas * nus_fac[:,None,None]
#     else:
#         nus = None
        
    #Yashar's AR noise plotting function
#     plot_AR_noise(Sigmas, nus, Plot_Dir=SaveDir, fname='-'.join(('AR_noise',fname_sffx))+'.pdf', SAVEFIG=opt['save'])

#     ##===== plot HMM parameters =====##
#     trans_matrix = np.exp(params_dict['transitions']['log_Ps'])
#     if opt['inputdriven'] & (opt['model_type'] == 'ARHMM_SL'):
#         Ws = params_dict['transitions']['Ws']
#     else:
#         Ws = None
    
    #Yashar's transition matrix plotting function
#     plot_trans_matrix(trans_matrix, state_usage, log_TM=False, errorbars=False, Ws=Ws, empirical=False, 
#                     Plot_Dir=SaveDir, fname='-'.join(('transmat_n_usage',fname_sffx))+'.pdf', SAVEFIG=opt['save'])
    
def save_plot_MAPseqs(SaveDir, fname_sffx, trsum, trMAPs, trPosteriors, 
                      trMasks, state_usage, opt, K, state_perm):
    
    if opt['save']:
        ioh5.save(os.path.join(SaveDir, '-'.join(('MAP_seqs',fname_sffx)) +'.h5'), 
                  {'trMAPs':trMAPs, 'trPosteriors':trPosteriors,'trMasks':trMasks, 
                    'state_usage':state_usage, 'hyperparams' : opt,'state_perm':state_perm})

    map_plots_fname = '-'.join(('MAP_seqs-AllConds',fname_sffx))+'.pdf'
    if not opt['save']:
        print('\n'+map_plots_fname+'\n')
    used_states = list(np.arange(K))
    
    #Yashar's MAP sequence plotting function
    plot_MAP_estimates(trMAPs, trMasks, used_states, trsum, opt, opt['mID'],
                             Plot_Dir=SaveDir, fname=map_plots_fname, SAVEFIG=opt['save'])    

    #TODO: currently durations are not plotted in all conditions 
    mapdurs_plots_fname = '-'.join(('MAP_state_durations-SomeConds',fname_sffx))+'.pdf'
    state_durations_list = map_state_durations(trMAPs, trMasks, K) 
    
    if not opt['save']:
        print('\n'+mapdurs_plots_fname+'\n')
    state_durations_list = [sdl[:-1] for sdl in state_durations_list] # getting rid of the durations of the NaN state
    
    #Yashar's State Duration plotting function
    plot_state_durations(state_durations_list, None, K, trsum, opt['mID'],
                             Plot_Dir=SaveDir, fname=mapdurs_plots_fname, SAVEFIG=opt['save'])
    
def save_plot_xval_lls(ll_training, ll_heldout,Ks, opt):

    xval_fname = '{}_lls_vs_K_{}to{}'.format(opt['model_type'],opt['Kmin'],opt['Kmax'])
    if opt['save']:
        ioh5.save(os.path.join(opt['SaveDirRoot'], xval_fname+'.h5'), {'ll_heldout':ll_heldout, 'll_training':ll_training,'Ks': Ks,
                                'RunTime':opt['RunTime'], 'kXval':opt['kXval'], 'Kmin':opt['Kmin'], 'Kmax':opt['Kmax']})
    #Plot Heldout log-likelihood
    rsi = 'RSI_{}{}{}'.format(opt['robust'],opt['sticky'],opt['inputdriven'])
    fname = '-'.join((xval_fname, 'heldout_ll', rsi))+'.pdf'
    plot_xval_lls_vs_K(ll_heldout, Ks,
            title_str='heldout log-likelihood', Plot_Dir=opt['SaveDirRoot'], fname=fname, SAVEFIG=opt['save'])   
    
    #Plot Training log-likelihood
    fname = '-'.join((xval_fname, 'training_ll', rsi))+'.pdf'
    plot_xval_lls_vs_K(ll_training, Ks,
            title_str='training log-likelihood', Plot_Dir=opt['SaveDirRoot'], fname=fname, SAVEFIG=opt['save'])   
        
