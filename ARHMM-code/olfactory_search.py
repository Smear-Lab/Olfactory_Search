#Misc
import os, time, argparse
import h5py, json
import glob, fnmatch
from tqdm import tqdm
import pdb

#Base
import numpy as np
import pandas as pd
import scipy.stats as st

#Plotting
import matplotlib
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec

#State-Space Modeling
import ssm
from pyhsmm.util.text import progprint_xrange
from pybasicbayes.distributions import Gaussian, AutoRegression
from autoregressive.models import ARWeakLimitStickyHDPHMM, ARHMM

#User Modules
from utilities import *
from plotting import *

##===================== Parse command line arguments=========================##
parser = argparse.ArgumentParser(description='ARHMM Mouse')

parser.add_argument('--Nmax', type=int, default=10,
                    help='max number of inferred states')

parser.add_argument('--A', type=float, default=1.0,
                    help='arhmm alpha')

parser.add_argument('--K', type=float, default=10000,
                    help='arhmm kappa')

parser.add_argument('--G', type=float, default=3.0,
                    help='arhmm gamma')

parser.add_argument('--S_0',type=float,default=1,
                    help='Observation Parameter for AR model')

parser.add_argument('--M_0',type=float,default=1,
                    help='Observation Parameter for AR model')

parser.add_argument('--N_samples', type=int, default=400,
                    help='number of iterations to run the Gibbs sampler')

parser.add_argument('--mouseID',type=str,default='all',
                    help='mouse to fit model to')

parser.add_argument('--cond', type=str, default='interleaved',
                    help='condition type')

parser.add_argument('--fit', type=int, default=1,
                    help='What iteration of fit is this')

parser.add_argument('--model', type=str, default='ARHMM',
                    help='ARHMM vs AR-HDP-HMM vs rARHMM')

parser.add_argument('--threshold', type=float, default=0.80,
                    help='MAP threshold')
#Directory Path of model parameter json file; not required if using other arguments
parser.add_argument('--json_dir', type=str)

args = parser.parse_args()

if __name__ == "__main__":
    ##=======================================================================##
    ##=========================  Hyper Parameters ===========================##
    startTime = time.time()
    # Later fraction of Gibbs samples to use for MAP
    fG = 0.50
    # x_t+1 = A x_t + b
    affine = True
    Nmax = args.Nmax
    # Number of Gibbs samples
    nGibbs = args.N_samples
    mouseID = args.mouseID
    # What condition are we analyzing?
    condition = args.cond
    alpha = args.A
    kappa = args.K
    gamma = args.G
    S_0 = args.S_0
    M_0 = args.M_0
    # How are we going to format the input data?
    MAP_threshold = args.threshold
    fitnum = args.fit
    model_type = args.model
    
    #Create folder for results    
    rDir = './results/{}'.format(model_type)
    # Check to see if directory exists
    if not os.path.isdir(rDir):
        os.makedirs(rDir)
    timestr = time.strftime('%Y%m%d_%H%M')
    #If this is another fit, save the results in another folder
    previous_fits = [d for d in os.listdir(rDir) if fnmatch.fnmatch(d,'Nmax*')]
    #Check to see if there are any results folders
    if len(previous_fits) == 0:
        if model_type == 'AR-HDP-HMM':
            foldername = 'Nmax{:2.0f}_{}_K{:.0E}G{:.0f}_Fit{:02d}'.format(Nmax,mouseID,kappa,gamma,fitnum)
            results_fname = '{}_results_K{:.0E}G{:.0f}_{}.h5'.format(model_type,kappa,gamma,timestr)
        elif model_type == 'rARHMM':
            foldername = 'Nmax{:02d}_{}_Fit{:02d}'.format(Nmax,mouseID,fitnum)
            results_fname = '{}_results_N{}_{}.h5'.format(model_type,Nmax,timestr)
        else:
            foldername = 'Nmax{:02d}_{}_Fit{:02d}'.format(Nmax,mouseID,fitnum)
            results_fname = '{}_results_N{}_{}.h5'.format(model_type,Nmax,timestr)
    else:
        foldername = previous_fits[0]
        while foldername in previous_fits:
            if model_type == 'AR-HDP-HMM':
                foldername = 'Nmax{:2.0f}_{}_K{:.0E}G{:.0f}_Fit{:02d}'.format(Nmax,mouseID,kappa,gamma,fitnum)
                results_fname = '{}_results_K{:.0E}G{:.0f}_{}.h5'.format(model_type,kappa,gamma,timestr)
            elif model_type == 'rARHMM':
                foldername = 'Nmax{:02d}_{}_Fit{:02d}'.format(Nmax,mouseID,fitnum)
                results_fname = '{}_results_N{}_{}.h5'.format(model_type,Nmax,timestr)
            else:
                foldername = 'Nmax{:02d}_{}_Fit{:02d}'.format(Nmax,mouseID,fitnum)
                results_fname = '{}_results_N{}_{}.h5'.format(model_type,Nmax,timestr)
            fitnum+=1

    SaveDir = os.path.join(rDir,foldername)
    # Check to see if directory exists
    if not os.path.isdir(SaveDir):
        os.makedirs(SaveDir)

    #Save Hyperparameters in JSON file
    jsDict = args.__dict__
    jsDict['json_dir'] = SaveDir
    jsDict.update(ModelStartTime = startTime)
    js_fname = 'ModelParameters.json'

    with open(os.path.join(SaveDir,'ModelParameters.json'),'w') as jsfile:
        json.dump(jsDict,jsfile,indent=4)

    ##======================  Read in Observation data =======================##
    DataDir = './data'
    total_trials = 0
    data_fulllist = []
    angle_fulllist = []
    tvec_fulllist = []
    trsum_list = []
    trial_lengths = []
        
    if mouseID == 'all':
        MouseIDs = sorted([d for d in os.listdir(DataDir) if fnmatch.fnmatch(d,'Mouse*')])

        #Read in data for each mouse and condition
        for mID in MouseIDs:
            condlist = os.listdir(os.path.join(DataDir,mID))
            for cond in condlist: 
                data_mouse, angles_m, tvec_m, trsum_m, nTrials_mouse, trial_lengths_m = read_trial_data(MouseID=mID,Condition=cond)
                
                #Append data to full list
                data_fulllist.extend(data_mouse)
                angle_fulllist.extend(angles_m)
                tvec_fulllist.extend(tvec_m)
                trsum_list.append(trsum_m)
                trial_lengths.extend(trial_lengths_m)
                total_trials += nTrials_mouse
        
    else:
        MouseIDs = [mouseID]
        condlist = os.listdir(os.path.join(DataDir,mouseID))
        
        for cond in condlist: 
            data_mouse, angles_m, tvec_m, trsum_m, nTrials_mouse, trial_lengths_m = read_trial_data(MouseID=mouseID,Condition=cond)

            #Append data to full list
            data_fulllist.extend(data_mouse)
            angle_fulllist.extend(angles_m)
            tvec_fulllist.extend(tvec_m)
            trsum_list.append(trsum_m)
            trial_lengths.extend(trial_lengths_m)
            total_trials += nTrials_mouse

    # Concatenate session trial summary data frames into 1 dataframe
    trsum_all = pd.concat(trsum_list)
 
    print('Data Read in')
    #Print Summary Message
    ptile = .95
    tlens_sorted = sorted(trial_lengths)
    percentile_index = int(np.ceil(ptile*total_trials))
    maxNumFrames = tlens_sorted[percentile_index]

    #Find the indices that correspond to trials longer than the 90th percentile
    short_trial_indices = np.where(np.array(trial_lengths) < maxNumFrames)[0]

    #Take only these trials for analysis
    trsum = trsum_all.take(short_trial_indices)
    trsum.reset_index(drop=True,inplace=True)
    data_tuples = [trtuple for index,trtuple in enumerate(data_fulllist) if index in short_trial_indices]
    tvec_list = [tv for index,tv in enumerate(tvec_fulllist) if index in short_trial_indices]
    angle_list = [ang for index,ang in enumerate(angle_fulllist) if index in short_trial_indices]

    # Update total Trials
    nTrials = len(data_tuples)

    data_list = [d[-1] for d in data_tuples]
    # Number of obserations per time step
    D_obs = data_tuples[0][-1].shape[1]
    
    #Save which trials are going through the ARHMM 
    trsum.to_csv(os.path.join(SaveDir,'inputted_trials.txt'),header=False,index=False,sep='\t',float_format='%.4f')     
    print('{} Total Trials'.format(nTrials))
    
    #Save the data as well 
    # fname = 'inputted_data_N{}_{}.h5'.format(Nmax,timestr)
    # with h5py.File(os.path.join(SaveDir,fname),'w') as h5file:
    #     gData = h5file.create_group('data_list')  
    #     for iTrial,data in enumerate(data_list):
    #         gData.create_dataset(name=str(iTrial),data=data)
    # pdb.set_trace()      
    ##======================= Initialize ARHMM object =======================##
    dynamics_hypparams = \
        dict(nu_0=D_obs + 2,
             S_0=S_0*np.eye(D_obs),
             M_0=M_0*np.hstack((np.eye(D_obs), np.zeros((D_obs,int(affine))))),
             K_0=np.eye(D_obs + affine),
             affine=affine)

    # Initialize a list of autorgressive objects given the size of the
    # observations and number of max discrete states
    dynamics_distns = [AutoRegression(A=np.column_stack((0.99 * np.eye(D_obs),np.zeros((D_obs, int(affine))))),sigma=np.eye(D_obs),**dynamics_hypparams) for _ in range(Nmax)]

    init_distn = Gaussian(nu_0=D_obs + 2,
                          sigma_0=np.eye(D_obs),
                          mu_0=np.zeros(D_obs),
                          kappa_0=1.0)    
    
    #What type of ARHMM model do we want to use?
    if model_type == 'AR-HDP-HMM':
        print('AR WeakLimit Sticky HDPHMM')
        # Create AR-HDP-HMM Object
        arhmm = ARWeakLimitStickyHDPHMM(
            init_state_distn='uniform',
            init_emission_distn=init_distn,
            obs_distns=dynamics_distns,
            alpha=alpha, kappa=kappa, gamma=gamma)
    elif model_type == 'rARHMM':
        print('recurrent ARHMM')
        arhmm = PGRecurrentARHMM(trans_params=dict(sigmasq_A=10000., sigmasq_b=10000.),
                init_state_distn='uniform',
                obs_distns=dynamics_distns,
                D_in = D_obs) 
    else:
        print('Vanilla ARHMM')
        arhmm = ARHMM(
            alpha=4.,
            init_emission_distn=init_distn,
            init_state_distn='uniform',
            obs_distns=dynamics_distns)
   
    ##===================== Add data to the ARHMM model======================##
    print("Adding data to {}".format(model_type))
    for trtuple in data_tuples:
        # Add data per trial
        arhmm.add_data(trtuple[-1])
    #pdb.set_trace()
    #arhmm.trans_distn.get_trans_matrices(trtuple[-1][0:1])
    #if model_type == 'rARHMM':
     #   print("Initializing dynamics with Gibbs sampling")
      #  for itr in tqdm(range(100)):
       #     arhmm.resample_obs_distns()
    #pdb.set_trace()
    ##====================== Sample from the ARHMM model=====================##
    # Create a list of lists to hold state sequence samples & the AR matrices
    stateseq_smpls = [[] for i in range(nTrials)]
    A_smpls = [[] for i in range(Nmax)]
    sigma_smpls = [[] for i in range(Nmax)]
    trans_matrix_smpls = []
    #pdb.set_trace()
    print("Fitting {}".format(model_type))
    # Loop over samples
    for iSample in progprint_xrange(nGibbs):
        # Sample Model
        arhmm.resample_model()

        # Append each Gibbs sample for each trial
        for iTrial in range(len(arhmm.states_list)):
            stateseq_smpls[iTrial].append(arhmm.states_list[iTrial].stateseq.copy())

        # Append the ARHMM matrix A and transition matrix for this sample
        for state in range(Nmax):
            A_smpls[state].append(arhmm.obs_distns[state].A.copy())
            sigma_smpls[state].append(arhmm.obs_distns[state].sigma.copy())
            
        if model_type != 'rARHMM':
            trans_matrix_smpls.append(arhmm.trans_distn.trans_matrix.copy())

    # Calculate the mean A, B, and transition matrix for all
    As_mean = [np.mean(Ai[int(fG*nGibbs):],axis=0) for Ai in A_smpls]
    sigma_mean = [np.mean(sig[int(fG*nGibbs):],axis=0) for sig in sigma_smpls]
    if model_type == 'rARHMM':
        trans_matrix_mean = []
    else:
        trans_matrix_mean = np.mean(trans_matrix_smpls[int(fG*nGibbs):],axis=0)
        
    # How many states were discovered?
    used_states = sorted(arhmm.used_states)
    dis_k = len(used_states)
    print('{} States discovered out of a max of {}'.format(dis_k,Nmax))
    used_states_sorted = [x for _, x in sorted(zip(arhmm.state_usages,used_states),reverse=True, key=lambda pair: pair[0])]
    
    #Save the log likelihood
    llhood = arhmm.log_likelihood()
    print('\nLog Likelihood: {}'.format(llhood))
    
    ##=================== Construct ARHMM MAP estimate ======================##
    print('\nConstructing MAP estimates of the trial state sequences')
    trMAPs = []
    max_posterior_prob = []
    trMASKs = []
    
    #Loop over all trials and calculate the maximum a posteriori probability; 
    #i.e. the mode of the posterior distribution P(z|x) 
    for iTrial in range(nTrials):
        # Take second half of gibbs samples to construct MAP
        z_smpls = np.array(stateseq_smpls[iTrial][int(fG*nGibbs):])
        
        state_probs_trial = []
        for state in used_states:
            state_occurances = np.isin(z_smpls,state)
            try:
                state_probs_trial.append(np.sum(state_occurances,axis=0)/z_smpls.shape[0])
            except:
                state_probs_trial.append([])
                print('Error??')
        #pdb.set_trace()
          
        #Save the maximum posterior probability for each time step
        pprob = np.max(np.hstack((np.zeros((len(used_states),1)),np.array(state_probs_trial))),axis=0)
        max_posterior_prob.append(pprob)
        mask = pprob < MAP_threshold
        trMASKs.append(mask)
        
        #Use the maximum posterior probability to determine a robust MAP State sequence
        MAP = np.concatenate(([np.nan],np.ndarray.flatten(stats.mode(z_smpls)[0])))

        #Add MAP to list           
        trMAPs.append(MAP)
    
    ##======================== Save ARHMM results ===========================##
    #Write important data for later analysis
    with h5py.File(os.path.join(SaveDir,results_fname),'w') as h5file:
        #Save calculated AR matrices and transition matrix
        h5file.create_dataset('ABs',data=As_mean)
        h5file.create_dataset('Sigma_s',data=sigma_mean)
        h5file.create_dataset('trans_matrix_mean',data=trans_matrix_mean)
        h5file.create_dataset('used_states',data=used_states)
        h5file.create_dataset('used_states_sorted',data=used_states_sorted)
        h5file.create_dataset('log_likelihood',data=llhood)
        h5file.create_dataset('state_usage',data=arhmm.state_usages)
        
        #Save MAP state sequences generated from ARHMM
        gMAP = h5file.create_group('trMAPs')
        for iTrial,MAPseq in enumerate(trMAPs):
            gMAP.create_dataset(name=str(iTrial),data=MAPseq)
          
        #Save the posterior probabilities associated with the state sequence
        gPost = h5file.create_group('max_posterior_prob')
        for iTrial,pp in enumerate(max_posterior_prob):
            gPost.create_dataset(name=str(iTrial),data=pp)

        #Save ARHMM Parameters s.t. the object can be recreated
        gObs_distns = h5file.create_group('obs_distns')
        for s,obs_distn in enumerate(arhmm.obs_distns):
            gObs_distns.create_dataset(name='sigma_{}'.format(s),data=arhmm.obs_distns[s].sigma)
            gObs_distns.create_dataset(name='AB_{}'.format(s),data=arhmm.obs_distns[s].A)
            #gObs_distns.create_dataset(name='natural_hypparam_{}'.format(s),data=arhmm.obs_distns[s].natural_hypparam)
            #gObs_distns.create_dataset(name='mf_natural_hypparam_{}'.format(s),data=arhmm.obs_distns[s].mf_natural_hypparam)
        
        #Save the transition parameters
        if model_type != 'rARHMM':
            gTrans_distns = h5file.create_group('trans_distns')
            #gTrans_distns.create_dataset(name='exp_expected_log_trans_matrix',data=arhmm.trans_distn.exp_expected_log_trans_matrix)
            gTrans_distns.create_dataset(name='trans_matrix',data=arhmm.trans_distn.trans_matrix)
        #gTrans_distns.create_dataset(name='alpha',data=arhmm.trans_distn.alpha)
        #gTrans_distns.create_dataset(name='alphav',data=arhmm.trans_distn.alphav)

    ##============================ Plotting ==================================##
    for mID in MouseIDs:
        mouse_mask = np.array(trsum['mID']==mID,dtype=bool)
        trsum_m = trsum[mouse_mask]
        
        #Take subset of lists for this particular mouse
        trMASKs_m = [m for m,b in zip(trMASKs,mouse_mask) if b]
        trMAPs_m = [m for m,b in zip(trMAPs,mouse_mask) if b]       
        plot_MAP_estimates(trMAPs_m,trMASKs_m,used_states_sorted,trsum_m,jsDict,mID,Plot_Dir=SaveDir)


    plot_model_convergence(stateseq_smpls,A_smpls,trans_matrix_smpls,jsDict,Plot_Dir=SaveDir)
    plot_arhmm_parameters(As_mean,jsDict,Plot_Dir=SaveDir)
    
    xstar = []
    for s,AB in enumerate(As_mean):
        A = AB[:,:-1]
        B = AB[:,-1]
        xstar.append(np.matmul(np.linalg.inv(np.eye(D_obs)-A),B))
    plot_fixed_points(As_mean,xstar,used_states,used_states_sorted,jsDict,Plot_Dir=SaveDir)