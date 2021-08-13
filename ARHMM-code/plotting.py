#Misc
import time, os, sys, pdb
from glob import glob
from fnmatch import fnmatch

#Base
import numpy as np
import pandas as pd

#Save
import json
import scipy.io as sio
import h5py

#User
from utilities import *

#Plot
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages

#Colors!
color_names=['windows blue','red','amber','faded green','dusty purple','orange','steel blue','pink','greyish',
             'mint','clay','light cyan','forest green','pastel purple','salmon','dark brown','lavender','pale green',
             'dark red','gold','dark teal','rust','fuchsia','pale orange','cobalt blue','mahogany','cloudy blue',
             'dark pastel green','dust','electric lime','fresh green','light eggplant','nasty green']

color_palette = sns.xkcd_palette(color_names)
cc = sns.xkcd_palette(color_names)
sns.set_style("darkgrid")
sns.set_context("notebook")

#-------------------------------------------------------------------------------
# Dictionaries for decoding trial summary data
Cond_Dict = {0:'100-0', 1:'80-20', 2:'60-40', 3:'Control', 4:'1% Abs-Conc',5:'0.1% Abs-Conc'}
Port_Dict = {2:'Left', 1:'Right'}
Resp_Dict = {1:'Correct',0:'Incorrect'}
Turn_Dict = {2:'Left', 1:'Right'}
Cond_InvDict = {'100-0':0, '80-20':1, '60-40':2, 'Control':3, '1% Abs-Conc':4, '0.1% Abs-Conc':5}
Port_InvDict = {'Left':2, 'Right':1}
Resp_InvDict = {'Correct':1,'Incorrect':0}

Plot_Dir = './plots/'

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
    mask = np.zeros(len(colors), dtype=bool)
    mask[used_states] = True
    
    color_names_shortened = [cn for cn, s in zip(color_names,mask) if s]
    color_names_shuffled = [x for _, x in sorted(zip(used_states,color_names_shortened), key=lambda pair: pair[0])]
    colors_shf = sns.xkcd_palette(color_names_shuffled)
    cc = [rgb for rgb in colors_shf]
    
    return cc,colors_shf

#-------------------------------------------------------------------------------
def get_colors(N_used_states):
    names = color_names[:N_used_states]
    colors_out = sns.xkcd_palette(names)
    cc = [rgb for rgb in colors_out]

    return cc, colors_out

#-------------------------------------------------------------------------------
def plot_z_samples(zs,used_states,xmax=None,
                   plt_slice=None,
                   N_iters=None,
                   title=None,
                   ax=None,
                   pdf=None):
    if ax is None:
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111)

    zs = np.array(zs)
    if plt_slice is None:
        plt_slice = (0, zs.shape[1])
    if N_iters is None:
        N_iters = zs.shape[0]

    # How many states were discovered?
    K = len(used_states)

    #create a mask for the correct color-map
    #mask = np.zeros(len(colors), dtype=bool)
    #mask[used_states] = True
    #cc = [rgb for rgb, s in zip(colors,mask) if s]

    cc, colors_shf = shuffle_colors(used_states)
    
    # Plot StateSeq as a heatmap
    im = ax.imshow(zs[:, slice(*plt_slice)], aspect='auto', vmin=0, vmax=K - 1,
                   cmap=gradient_cmap(color_palette), interpolation="nearest",
                   extent=plt_slice + (N_iters, 0))

    # Create a legend
    # get the colors of the values, according to the colormap used by imshow
    # create a patch (proxy artist) for every color
    patches = [mpatches.Patch(color=colors[int(s)], label="State {:.0f}".format(s)) for s in used_states]
    # put those patched as legend-handles into the legend
    plt.legend(handles=patches, loc=4)

    ax.set_ylabel("Trial")
    ax.set_xlabel("Frame #")
    if xmax is not None:
        plt.xlim(0,xmax)

    if title is not None:
        ax.set_title(title)

    if pdf is not None:
        pdf.savefig(fig)

    # Close the figures
    plt.close('all')

#-------------------------------------------------------------------------------
def plot_MAP_estimates(trMAPs,trMASKs,used_states,trsum,args,mouseID,apply_mask=True,Plot_Dir=None,fname=None):
    # Get number of trials from trial summary data frame
    nTrials = len(trsum)
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

    #Apply mask to MAP state sequence, which is a copy with local scope
    if apply_mask:
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
                indy = np.where((trsum['port'] == lr) & (trsum['cond'] == iCond) & (trsum['resp'] == resp))
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
                    # End of Port_Dict loop
                # End of Cond_Dict loop
            # End of Resp_Dict loop
    # Close PDF file
    pdfdoc.close()

#-------------------------------------------------------------------------------
def plot_trans_matrix(trans_matrix,state_usages,dis_k,title=None,pdf=None):

    # Convert numpy arrays into Panda DataFrames
    tm = pd.DataFrame(np.log(trans_matrix))
    su = pd.DataFrame(state_usages)

    # Plotting Properties
    fig = plt.figure(figsize=(10,5))
    gs = gridspec.GridSpec(1,2)
    fp = FontProperties()
    fp.set_weight("bold")
    if len(state_usages)>dis_k:
        # Calculate error bars for plot
        ci = 'sd'
    else:
        # Only 1 array of state usages
        ci = None

    # Draw a heatmap with the numeric values in each cell
    cmap = sns.cubehelix_palette(light=1, as_cmap=True)
    ax1 = fig.add_subplot(gs[0,0])
    sns.heatmap(tm, cmap=cmap,annot=True, fmt=".2f",vmin=0, vmax=1, linewidths=.5, ax=ax1,square=True,cbar_kws={'label': 'Probability'})
    ax1.set_title('log(Transition Probability Matrix)')

    # Plot overall state_usages
    colors = sns.xkcd_palette(color_names)
    ax2 = fig.add_subplot(gs[0,1])
    ax2 = sns.barplot(data =su,ci = ci,orient='h',palette=colors)
    ax2.set_xlabel('Probability')
    ax2.set_ylabel('State')
    ax2.set_title('State Usage')

    # Set Super Title
    if title is not None:
        fig.suptitle(title)
    else:
        fig.suptitle('Overall ARHMM Fit')

    if pdf is not None:
        pdf.savefig(fig)

    #Close figures
    plt.close('all')
    
def construct_trans_matrices(arhmm,trMAPs,trans_matrix_mean,trsum,args,Plot_Dir=None):

    nTrials = len(trsum)
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

                # Find the indices of the trsum that correspond to the condition
                indy = np.where((trsum['cond'] == iCond) & (trsum['port'] == lr) & (trsum['resp'] == resp))
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
def plot_model_convergence(stateseq_smpls,ABs,trans_matrices,args,Plot_Dir=None,fname=None):

    # Create a pdf document for plots
    timestr = time.strftime('%Y%m%d_%H%M')
    if Plot_Dir is None:
        Plot_Dir = './plots'
    
    # Create a pdf document for plots
    if fname is None:
        if args['model'] is 'AR-HDP-HMM':
            fname = 'arhmm_Convergence_A{:.0f}_K{:.0f}G{:.0f}_{}.pdf'.format(args['A'],args['K'],args['G'],timestr)
        else:
            fname = 'arhmm_Convergence_N{}_{}.pdf'.format(args['Nmax'],timestr)

    pdfdoc = PdfPages(os.path.join(Plot_Dir,fname))

    # Get some shapes
    nTrials = len(stateseq_smpls)
    D_obs = ABs[0][0].shape[0]
    d2 = D_obs**2
    num_states = len(ABs)
    n2 = num_states**2
    fdim = D_obs*(D_obs+1)
    nGibbs = len(ABs[0])

    # Loop over each trial & calculate the number of state transitions between
    # Gibbs samples; normalize to the trial length
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)

    stateseq_num_trans = []
    for iTrial in random.sample(range(nTrials),100):
        tr_smpls = stateseq_smpls[iTrial]
        tr_trans = []
        nBins = tr_smpls[0].shape[0]
        for iSmpl in range(nGibbs-1):
            tr_trans.append(sum(tr_smpls[iSmpl] != tr_smpls[iSmpl+1])/nBins)
        plt.plot(tr_trans)
        # Append to the overall trial list
        stateseq_num_trans.append(tr_trans)
    plt.plot(np.mean(stateseq_num_trans,axis=0),'-k',LineWidth=5)
    ax.set_xlabel('Gibbs Sample Iteration')
    ax.set_ylabel('Number of State Transitions')
    ax.set_title('ARHMM State Transitions')
    pdfdoc.savefig(fig)

    # Loop over the different states
    for state,ABi in enumerate(ABs):
        AB_raster = np.empty((fdim,nGibbs))

        # Loop over Gibbs samples of each state matrix A and flatten
        for ii,sampleAB in enumerate(ABi):
            AB_raster[:,ii] = sampleAB.flatten('F').copy()

        # Plot convergence of model parameters
        fig = plt.figure(figsize=(10,5))
        gs = gridspec.GridSpec(1,2)
        ax1 = fig.add_subplot(gs[0,0])
        for iRow in range(d2):
            plt.plot(AB_raster[iRow,:])
        ax1.set_xlabel('Gibbs Sample Iteration')
        ax1.set_ylabel('Model Parameter Value')
        ax1.set_title('Elements of A')

        ax2 = fig.add_subplot(gs[0,1])
        for iRow in range(d2,fdim):
            plt.plot(AB_raster[iRow,:])
        ax2.set_xlabel('Gibbs Sample Iteration')
        ax2.set_ylabel('Model Parameter Value')
        ax2.set_title('Elements of B')
        fig.suptitle('Model Parameter Convergence for state {}'.format(state))
        # Save figure
        pdfdoc.savefig(fig)
        # Close figure
        plt.close(fig)

    # Plot how the transition matrix changes over Gibbs samples
    T_raster = np.empty((n2,nGibbs))
    for ii,sampleT in enumerate(trans_matrices):
        T_raster[:,ii] = sampleT.flatten('F').copy()

    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)
    for iRow in range(n2):
        plt.plot(T_raster[iRow,:])
    ax.set_xlabel('Gibbs Sample Iteration')
    ax.set_ylabel('Transition Matrix Element Value')
    ax.set_title('Transition Matrix convergence')
    # Save figure
    pdfdoc.savefig(fig)
    # Close figure
    plt.close(fig)

    # Close PDF file
    pdfdoc.close()

    #-------------------------------------------------------------------------------
def plot_arhmm_parameters(ABs,args,Plot_Dir=None):

    # Create a pdf document for plots
    timestr = time.strftime('%Y%m%d_%H%M')
    if Plot_Dir is None:
        Plot_Dir = './plots'

    # Create a pdf document for plots
    if args['model'] is 'AR-HDP-HMM':
        fname = 'arhmm_Parameters_A{:.0f}_K{:.0f}G{:.0f}_{}.pdf'.format(args['A'],args['K'],args['G'],timestr)
    else:
        fname = 'arhmm_Parameters_N{}_{}.pdf'.format(args['Nmax'],timestr)

    pdfdoc = PdfPages(os.path.join(Plot_Dir,fname))

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

        # Plot Eigenvalues of A
        ax1 = fig.add_subplot(gs[0,0])
        plt.plot(np.real(w),np.imag(w),'.k',MarkerSize=9)
        ax1.add_patch(plt.Circle((0, 0), radius=1, edgecolor='b', facecolor='None',LineWidth=2))
        # ax1.set_aspect('equal', 'box')
        plt.axis('equal')
        ax1.set_xlabel('Real')
        ax1.set_ylabel('Imag')
        ax1.set_title('Eigenvalues of A')
        ax1.grid()

        # Plot ARHMM matrix A
        ax2 = fig.add_subplot(gs[0,1:4])
        sns.heatmap(A-np.eye(D_obs), cmap=cmap,annot=True, fmt=".2f", linewidths=.5, ax=ax2,square=True,cbar=False)
        ax2.set_title('A Matrix')

        ax3 = fig.add_subplot(gs[0,4])
        sns.heatmap(xstar, annot=True, fmt=".2f", linewidths=.5, ax=ax3,cbar=False,square=True)
        ax3.set_title('x*')

        fig.suptitle('ARHMM Parameters for state {}'.format(state))
        pdfdoc.savefig(fig)

    # Close PDF file
    pdfdoc.close()

    #Close figures
    plt.close('all')

#-------------------------------------------------------------------------------
def plot_arena(data=None,xstar=None,used_states = None,ax=None,fig=None):
    fig = fig if fig else plt.figure(figsize=(10,5))
    ax = ax if ax else fig.add_subplot(111)
    ax.set_axis_off()
    if data is not None:
        for i in range(0,data.shape[1],2):
            ci = int((i%data.shape[1])/2)
            plt.scatter(data[:,i],data[:,i+1],c=color_palette[ci],s=10)
        # plt.scatter(data[:,0],data[:,1],c=colors[0],s=10)
        # plt.scatter(data[:,2],data[:,3],c=colors[1],s=10)
        # plt.scatter(data[:,4],data[:,5],c=colors[2],s=10)

    # Plot the boundaries of the arena
    plt.plot([80,1104],[44,44],'-k',LineWidth = 3,zorder=1)
    plt.plot([80,1104],[663,663],'-k',LineWidth = 3,zorder=1)
    plt.plot([80,80],[44,663],'-k',LineWidth = 3,zorder=1)
    plt.plot([1104,1104],[44,663],'-k',LineWidth = 3,zorder=1)
    plt.plot([421,421],[44,663],'--k',LineWidth = 3,zorder=1)
    plt.plot([80,421],[354,354],'--k',LineWidth = 3,zorder=1)
    plt.plot(1127,349,'ok',MarkerSize = 10)
    plt.plot(201,675,'sk',MarkerSize = 10)
    plt.plot(173,24,'sk',MarkerSize = 10)
    plt.xlim([-620,1820])
    plt.ylim([-370,1070])
    plt.xticks([])
    plt.yticks([])
    plt.axis('equal')
    
    #Plot circle
    NH = 100
    xvec = np.linspace(1057,1127,100)
    pys = 350 + np.sqrt(NH**2 - (1127-xvec)**2)
    nys = 350 - np.sqrt(NH**2 - (1127-xvec)**2)
    plt.plot(xvec,pys,'-k')
    plt.plot(xvec,nys,'-k')
    
    if xstar is not None:
        xl = [-600,1800]
        yl = [-350,1050]
        ac = [592,354]
        for state, x in zip(used_states,xstar):
            # If its within a certain bound of the arena, plot the fixed point
            if (xl[0] <= x[2] <= xl[1]) and (yl[0] <= x[3] <= yl[1]):
                plt.scatter(x[2],x[3],c=colors[state],edgecolors='k',s=100,marker=(5,1),zorder=2)
            else:
            # If not, plot it at a boundary
                m = (x[3]-ac[1])/(x[2]-ac[0])
                b = x[3]-m*x[2]

                #If x value is within bounds & the y-value is above/below plot
                if (xl[0] <= x[2] <= xl[1]):
                    plt.scatter(x[2],m*x[2]+b,c=colors[state],edgecolors='k',s=100,marker=(5,1),zorder=2)
                #If the x-value is right of plot
                elif (x[2] > xl[1]):
                    y=m*xl[1]+b
                    if y > yl[1]:
                        y=yl[1]
                    elif y < yl[0]:
                        y=yl[0]

                    plt.scatter(xl[1],y,c=colors[state],edgecolors='k',s=100,marker=(5,1),zorder=2)
                #If the x-value is left of plot
                elif (x[2] < xl[0]):
                    y=m*xl[1]+b
                    if y > yl[1]:
                        y=yl[1]
                    elif y < yl[0]:
                        y=yl[0]
                    plt.scatter(xl[0],y,c=colors[state],edgecolors='k',s=100,marker=(5,1),zorder=2)

        plt.xlim([-620,1820])
        plt.ylim([-370,1070])
        patches = [mpatches.Patch(color=colors[s], label="State {:.0f} : ({:.0f},{:.0f}) ".format(s,x[2],x[3])) for s,x in zip(used_states,xstar)]
        # put those patched as legend-handles into the legend
        plt.legend(handles=patches, loc=4)
        ax.set_title('ARHMM Fixed Points')

#-------------------------------------------------------------------------------
def plot_fixed_points(As_mean,xstar,used_states,used_states_sorted,args,Plot_Dir=None):
    
    #Get Plot_Dir to save plots
    timestr = time.strftime('%Y%m%d_%H%M')
    if Plot_Dir is None:
        Plot_Dir = './plots'
    
    cc, colors_shf = shuffle_colors(used_states_sorted)
        
     # Create a pdf document for plots
    if args['model'] is 'AR-HDP-HMM':
        fname = 'FixedPoints_K{:.0f}G{:.0f}_{}.pdf'.format(args['K'],args['G'],timestr)
    else:
        fname = 'FixedPoints_N{}_{}.pdf'.format(args['Nmax'],timestr)
    pdfdoc = PdfPages(os.path.join(Plot_Dir,fname))
    
    #Plot fixed points
    sfig = plt.figure(figsize=(10,5))
    ax = sfig.add_subplot(111)
    #Plot Arena
    plot_arena(ax=ax,fig=sfig)

    #Loop over autoregressive matrices
    fpt_stability = []
    for i, (AB,x,s) in enumerate(zip(As_mean,xstar,used_states)):
        A = AB[:,:-1]
        B = AB[:,-1]

        #Get eigenvalues of A
        ww,vv = np.linalg.eig(A)

        #Sort based on absolute value
        sorted_eig = np.array(sorted(np.real(ww),key=abs,reverse=True))
        fpt = (sorted_eig % 1) - sorted_eig
        fpt_str=''
        for stability in fpt:
            if stability == -1 or stability == 2:
                fpt_str += '-'
            else:
                fpt_str += '+'
        fpt_stability.append(fpt_str)

        # plt.plot([x[0],x[2]],[x[1],x[3]],'-k')
        # plt.plot([x[2],x[4]],[x[3],x[5]],'-k')
        plt.scatter(x[2],x[3],c=colors_shf[s],s=40,marker='X', zorder=2)
        # plt.scatter(x[2],x[3],c=colors_shf[s],s=40,marker='H', zorder=2)
        # plt.scatter(x[4],x[5],c=colors_shf[s],s=40,marker='D', zorder=2)

    #Make sure limits are correct
    plt.xlim([-620,1820])
    plt.ylim([-370,1070])
    
    patches = [mpatches.Patch(color=colors_shf[s], label='State {:2d}: Fixed Pts: {}'.format(s,fp)) for s,fp in zip(used_states,fpt_stability)]
    # put those patched as legend-handles into the legend
    plt.legend(handles=patches, loc=4)
    ax.set_title('ARHMM Fixed Points')

    #Save Fig
    fpath = os.path.join(Plot_Dir,fname)
    sfig.savefig(fpath)
    # plt.show()
    plt.close('all')

#-------------------------------------------------------------------------------
def plot_angle_hist(trMAPs,angle_list,lr_list,used_states,used_states_sorted,args,mouseID,Plot_Dir=None):
    timestr = time.strftime('%Y%m%d_%H%M')
    if Plot_Dir is None:
        Plot_Dir = './plots'
    
    cc, colors_shf = shuffle_colors(used_states_sorted)
    
    # Create a pdf document for plots
>

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
def plot_state_durations(state_duration_list,used_states,used_states_sorted,trsum,args,mouseID,Plot_Dir=None):
    if Plot_Dir is None:
        Plot_Dir = './plots'
    timestr = time.strftime('%Y%m%d_%H%M')
    cc, colors_shf = shuffle_colors(used_states_sorted)
    nTrials = len(state_duration_list)
    Nmax = args['Nmax']
    
    # Create a pdf document for plots
    if args['model'] is 'AR-HDP-HMM':
        fname = 'State_Durations_{}_K{:.0f}G{:.0f}_{}.pdf'.format(mouseID,args['K'],args['G'],timestr)
    else:
        fname = 'State_Durations_{}_N{}_{}.pdf'.format(mouseID,args['Nmax'],timestr)
    pdfdoc = PdfPages(os.path.join(Plot_Dir,fname))
    
    
    #Group together all control trials, irrespective of response 
    fig = plt.figure(figsize=(10,5))
    gs = gridspec.GridSpec(2,len(used_states))
    indy_l = np.where((trsum['port'] == 2) & (trsum['cond'] != 3))[0]
    indy_r = np.where((trsum['port'] == 1) & (trsum['cond'] != 3))[0]
    fig.suptitle('State Durations for {} left and {} right non-control Trials'.format(len(indy_l),len(indy_r)))
    
    # Loop over active odor porth
    for lr in Port_Dict:
        # Find the indices of the trlist that correspond to the condition
        indy = np.where((trsum['port'] == lr) & (trsum['cond'] == 3))[0]
        s_dur = []
        for ii,state in enumerate(used_states):
            for iTrial in indy:
                s_dur.extend(state_duration_list[iTrial][ii])

            ax = fig.add_subplot(gs[lr-1,ii])
            sns.distplot(s_dur,color=colors[ii],ax=ax)
            ax.set_title('State {}'.format(state))
            ax.set_xlim([0,25])
            if ii == 0:
                ax.set_ylabel('{} Port Usage'.format(Port_Dict[lr]))
            else:
                ax.set_yticks([])
    pdfdoc.savefig(fig)

    ## ========= Plot state durations for various conditions =============# 
    #Loop over responses
    for resp in Resp_Dict:
        # Loop over conditions
        for iCond in Cond_Dict:
            #Make figure
            fig = plt.figure(figsize=(10,5))
            gs = gridspec.GridSpec(2,len(used_states))
                
            # Loop over active odor porth
            for lr in Port_Dict:
                # Find the indices of the trlist that correspond to the condition
                indy = np.where((trsum['port'] == lr) & (trsum['cond'] == iCond) & (trsum['resp'] == resp))[0]
            
                fig.suptitle('State Durations for {} {}, {} Trials'.format(len(indy),Resp_Dict[resp],Cond_Dict[iCond]))
                s_dur = []
                for ii,state in enumerate(used_states):
                    for iTrial in indy:
                        s_dur.extend(state_duration_list[iTrial][ii])

                    ax = fig.add_subplot(gs[lr-1,ii])
                    if len(s_dur) < 5:
                        continue
                    
                    sns.distplot(s_dur,color=colors_shf[state],ax=ax)
                    ax.set_title('State {}'.format(state))
                    ax.set_xlim([0,25])
                    if ii == 0:
                        ax.set_ylabel('{} Port Usage'.format(Port_Dict[lr]))
                    else:
                        ax.set_yticks([])
            #Save Figure
            pdfdoc.savefig(fig)
    pdfdoc.close()
    plt.close('all')
    
    #-------------------------------------------------------------------------------
def plot_AR_action(As_mean, used_states_sorted, state_usages, timesteps=(4,1),
                   ORDER_STATES=False, AXIS_OFF=False,
                   figsize=(15,6), rowcol=(2,5), args=None, Plot_Dir=None,
                   SAVEFIG=False, edgecolor='k'):

    #Get Plot_Dir to save plots
    if Plot_Dir is None:
        Plot_Dir = './plots'

    if SAVEFIG:
        timestr = time.strftime('%Y%m%d_%H%M')
         # Create a pdf document for plots
        if args['model'] is 'AR-HDP-HMM':
            fname = 'AR_Interpret_K{:.0f}G{:.0f}_{}.svg'.format(args['K'],args['G'],timestr)
        else:
            fname = 'AR_Interpret_N{}_{}.svg'.format(args['Nmax'],timestr)
        #pdfdoc = PdfPages(os.path.join(Plot_Dir,fname))


    state_usages = np.array(np.round(100*state_usages),dtype=np.int32)

    # if not ORDER_STATES:
    #     cc, colors_shf = shuffle_colors(used_states_sorted)
    # else:
    #     cc, colors_shf = shuffle_colors(used_states)
    #     As_mean = [As_mean[used_states_sorted[i]] for i in range(len(used_states)) ]

    cc, colors_shf = get_colors(len(used_states_sorted))

    if len(used_states_sorted)>rowcol[0]*rowcol[1]:
        rowcol = list(rowcol)
        rowcol[0] = 1 + len(used_states_sorted)/rowcol[1]

    step = timesteps[1]
    timesteps = timesteps[0]
    time_colors = np.linspace(0,1,timesteps+1).reshape((-1,1)) * np.ones((1,3))

    lNH = 65 #average nose-head distance
    lBH = 85 #average body-head distance
    lBN = lBH + lNH
    Bshift = 40
    xH, yH = 762, 350 #approximate average head coordinates
    xB, yB = 789, 354 #approximate average body coordinatesa
    x0s = np.array([[xB+Bshift,yB, xB+Bshift+lBH,yB, xB+Bshift+lBN,yB ,1],  #pointing towards initiation port
                    [xB-Bshift,yB, xB-Bshift-lBH,yB, xB-Bshift-lBN,yB ,1],  #pointing aways from initiation port
                    [xB,yB+Bshift, xB,yB+Bshift+lBH, xB,yB+Bshift+lBN ,1],  #pointing towards right wall
                    [xB,yB-Bshift, xB,yB-Bshift-lBH, xB,yB-Bshift-lBN ,1]]) #pointing towards left wall
    # #use this instead for Matt:
    #x0s = np.array([[xB-Bshift,yB, xB-Bshift-lBH,yB, xB-Bshift-lBN,yB ,1]])  #pointing aways from initiation port

    sfig = plt.figure(figsize=figsize)
    sfig.subplots_adjust(hspace=.4,wspace=.4)#.3)
    # #Plot Arena
    # plot_arena(ax=ax,fig=sfig, linecolor=edgecolor)
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
                #pdb.set_trace()
                x_old = x
                x = Ab @ x
                x = np.concatenate((x,[1]))

                if (t+1) % step ==0: #only plot every "step" steps
                    plotBHN(ax,x[:-1],t+1,sp)#,s)

        ax.axis('equal')
        ax.grid(False)
        # ax.set_xlim([450,800])
        # ax.set_ylim([200,550])
        # ax.set_xticks([500,600,700])
        # ax.set_yticks([300,400,500])
        # #ax.set_facecolor((1.,1.,1.))

        # s1 = s
        # if ORDER_STATES: s1 = used_states_sorted[s]
        # ax.set_title('state {}'.format(s1),color=colors_shf[s])

        ax.set_title('state {}, usage = {} %'.format(s,state_usages[s]),color=colors_shf[sp])

        # #Make sure limits are correct
        # plt.xlim([-620,1820])
        # plt.ylim([-370,1070])

    #Save Fig
    if SAVEFIG:
        fpath = os.path.join(Plot_Dir,fname)
        sfig.savefig(fpath)
        plt.close('all')
    else:
        plt.show()
        return (sfig,ax)
    
def plot_state_usage(used_states, state_usage_raw_list, trsum, Plot_Dir=None, SAVEFIG=False, Plot_Conds=False):

    #Get Plot_Dir to save plots
    if Plot_Dir is None:
        Plot_Dir = './plots'

    if SAVEFIG:
        timestr = time.strftime('%Y%m%d_%H%M')
        pos = Plot_Dir.find('Nmax')
        folder = Plot_Dir[pos:]
        fname2 = 'StateUsagePerCond_{}.svg'.format(folder)
        fname1 = 'StateUsageOverall_{}.svg'.format(folder)

       # pdfdoc = PdfPages(os.path.join(Plot_Dir,fname))
    #Various Parameters
    nTrials = len(trsum)
    Nmax = len(used_states)
    
    #Calculate the overall state usage
    state_usage_overall = np.sum(state_usage_raw_list,axis=0)/np.sum(state_usage_raw_list)
    used_states_sorted = [x for _, x in sorted(zip(state_usage_overall,used_states),reverse=True, key=lambda pair: pair[0])]

    # Plot overall state_usages
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    ax = sns.barplot(x = used_states,y = sorted(state_usage_overall,reverse=True), orient='v',palette=colors,edgecolor='k')
    ax.plot(used_states,np.ones(Nmax)*(1/Nmax),'--k')
    ax.set_xticklabels(used_states_sorted)
    ax.set_ylabel('Probability')
    ax.set_xlabel('State')
    ax.set_title('State Usage')
    if SAVEFIG:
        plt.savefig(os.path.join(Plot_Dir,fname1))
        plt.close()

    Conds_Dict = {0:'100-0', 1:'80-20', 2:'60-40', 4:'1% Abs-Conc',5:'0.1% Abs-Conc', 6:'90-30',7:'30-10',3:'Control'}
    fig = plt.figure(figsize=(28, 12))
    fig.suptitle('State Usage per Condition')
    gs = gridspec.GridSpec(2,8,wspace=0.05)
    #Loop over conditions
    for ii,iCond in enumerate(Conds_Dict):
        # Loop over active odor porth
        for jj,lr in enumerate(Port_Dict):
            mask = np.zeros(nTrials, dtype=bool)
            # Find the indices of the trlist that correspond to the condition
            indy = np.where((trsum['port'] == lr) & (trsum['cond'] == iCond) & (trsum['resp'] == 1))
            mask[indy] = True

            # Continue onto next condition if no trials exist
            if sum(mask) == 0:
                continue

            # Create a new list based of only that condition
            cond_SUs = [state_usage_raw_list[i] for i in range(nTrials) if mask[i]]
            state_usage = np.sum(cond_SUs,axis=0)/np.sum(cond_SUs)
            state_usage_sorted = [x for _, x in sorted(zip(state_usage_overall,state_usage),reverse=True, key=lambda pair: pair[0])]

            #fig = plt.figure(figsize=(10, 5))
            ax = fig.add_subplot(gs[jj,ii])
            ax = sns.barplot(x = used_states,y = state_usage_sorted, orient='v',palette=colors,edgecolor='k')
            ax.plot(used_states,np.ones(Nmax)*(1/Nmax),'--k')
            ax.set_xticklabels(used_states_sorted)
            ax.set_ylim([0,0.25])
            if ii == 0:
                ax.set_ylabel('Probability')
            else:
                ax.set_yticklabels([])
            if jj == 1:
                ax.set_xlabel('State')
            else:
                ax.set_xticklabels([])
            ax.set_title('{} {}'.format(Conds_Dict[iCond],Port_Dict[lr]))
    
    if SAVEFIG:
        plt.savefig(os.path.join(Plot_Dir,fname2))
        plt.close()
