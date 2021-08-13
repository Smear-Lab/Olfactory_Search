#Misc
import os, time, argparse
import h5py, json
import glob, fnmatch,pdb
from tqdm import tqdm
import multiprocessing
#Base
import numpy as np
import pandas as pd
import scipy.stats as st
from sklearn.model_selection import StratifiedKFold
#Plotting
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
#State-Space Modeling
#S.Linderman
import ssm
#M.Johnson
from pyhsmm.util.text import progprint_xrange
from pybasicbayes.distributions import Gaussian, AutoRegression
import autoregressive.models as pyhmm 
#User Modules
import utilities as util
import plotting_YAA as plots_YAA 

##===== Run Command =====##
# OMP_NUM_THREADS=1 python olfactory_search_xval.py --model_type "ARHMM_MJ" --Kmin 12 --Kmax 20 

##===== ============================ =====##
##===== Parse Command Line Arguments =====##
parser = argparse.ArgumentParser(description='ARHMM Mouse')

parser.add_argument('--save',type=bool, default=1,
                    help='Save Results?')
parser.add_argument('--json_dir', type=str, 
                    help='Directory Path of model parameter json file; not required if using other arguments')

##===== Data Options =====##
parser.add_argument('--mID',type=str, default='all_mice',
                    help='mouse to fit model to')
parser.add_argument('--condition', type=str, default='all_conds',
                    help='trial condition type')
parser.add_argument('--data_type', type=str, default='BHNx',
                    help='BHNx vs BHNxv vs EgoAllo_xv')
parser.add_argument('--HMM_inputs', type=str, default='BHNx',
                    help='BHNx vs BHNxv')
parser.add_argument('--x_units', type=str, default='pixels',
                    help='pixels or arena_length')

##===== Model Type =====##
parser.add_argument('--model_type', type=str, default='ARHMM_MJ',
                    help='ARHMM_SL or ARHMM_MJ')
parser.add_argument('--robust', type=bool, default=0,
                    help='autoregressive(0) or robust_autoregressive(1)')
parser.add_argument('--sticky', type=bool, default=0,
                    help='standard(0) or sticky(1) ARHMM')
parser.add_argument('--inputdriven', type=bool, default=0,
                    help='HMM transitions dependent on some input in addition to previous HMM state')

##===== Model Parameters =====##
parser.add_argument('--kappa', type=float, default=1e5,
                    help='sticky arhmm kappa')
parser.add_argument('--AR_lags', type=str, default=1,
                    help='Autoregressive lags')
parser.add_argument('--l2_penalty_A', type=float, default=0,
                    help='AR l2_penalty_A')
parser.add_argument('--l2_penalty_b', type=float, default=0,
                    help='AR l2_penalty_b')
parser.add_argument('--l2_penalty_V', type=float, default=0,
                    help='AR l2_penalty_V')
parser.add_argument('--MAP_threshold', type=float, default=0.80,
                    help='MAP threshold')
parser.add_argument('--nGibbs', type=int, default=200,
                    help='number of iterations to run the Gibbs sampler')
parser.add_argument('--burn_fraction', type=float, default=0.66,
                    help='Calculate MAP sequence with the last 37.5% of samples; of nGibbs = 400, 250 samples are burned')

##===== Run Options =====##
parser.add_argument('--Kmin', type=int, default=80,
                    help='minimum number of HMM states')
parser.add_argument('--Kmax', type=int, default=100,
                    help='maximum number of HMM states')
parser.add_argument('--kXval', type=int, default=5,
                    help='number of kfold')
parser.add_argument('--EM_tolerance', type=float, default=1e-5,
                    help='SSM EM algorithm tolerance')
parser.add_argument('--EM_iters', type=int, default=200,
                    help='EM Iterations')
parser.add_argument('--max_processes', type=int, default=18,
                    help='max # of parallel processes to run')
args = parser.parse_args()

def set_arhmm_hyperparams(opt,K):
    D_obs = opt['D_obs']
    Mobs = 0
    
    #Autoregressive keyword arguments
    ar_kwargs = dict(
            # l2_penalty_A= args_dic['l2_penalty_A'],
            # l2_penalty_b= args_dic['l2_penalty_b'],
            # l2_penalty_V= args_dic['l2_penalty_V'],
            lags = opt['AR_lags']
            )
    
    #HMM Transition parameters 
    trans_kwargs = dict(
            # alpha= args_dic['alpha'],
            )

    #Gaussian or t-distribution
    if not opt['robust']:
        observation_type = "autoregressive"
    else:
        observation_type = "robust_autoregressive"

    #What model are we going to run?
    if not opt['inputdriven']:
        M = 0
        if not opt['sticky']:
            if opt['model_type'] == 'ARHMM_MJ':
                print('Bayesian ARHMM')
            else:
                print('Vanilla ARHMM')
            transition_type = "standard"                
        else:
            print('sticky ARHMM') 
            transition_type = "sticky"        
            trans_kwargs['kappa'] = opt['kappa']
    else:
        M = D_obs
    #   trans_kwargs['l2_penalty'] = args_dic['l2_penalty_W'] #coeff of l2-regul penalty on W (weights of logistic regression)       
        transition_type = "inputdriven"        
        if not opt['sticky']:
            print('input-driven ARHMM')             
        else:
            print('input-driven sticky ARHMM') 
            trans_kwargs['kappa'] = opt['kappa']
    
    #If we're using matt Johnsons code, most of the above parameters don't matter
    #Initialize Observation distribution and set it to ar_kwargs
    if opt['model_type'] == 'ARHMM_MJ':
        affine = True
        dynamics_hypparams = \
            dict(nu_0=D_obs + 2,
                 S_0=np.eye(D_obs),
                 M_0=np.hstack((np.eye(D_obs), np.zeros((D_obs,int(affine))))),
                 K_0=np.eye(D_obs + affine),
                 affine=affine)

        # Initialize a list of autorgressive objects given the size of the
        # observations and number of max discrete states
        ar_kwargs = [AutoRegression(A=np.column_stack((0.99 * np.eye(D_obs),\
                            np.zeros((D_obs, int(affine))))),sigma=np.eye(D_obs),\
                                          **dynamics_hypparams) for _ in range(K)]
        
    return D_obs, M, Mobs, observation_type, ar_kwargs, transition_type, trans_kwargs

def make_hyperparams_dic(opt, K, M, trans_kwargs, ar_kwargs):
    hyperparams = opt.copy()
    del hyperparams['Kmin'], hyperparams['Kmax']
    hyperparams['K'] = K
    hyperparams['M'] = M        
#     hyperparams['Mobs'] = Mobs        
    hyperparams['trans_kwargs'] = trans_kwargs 
    if opt['model_type'] == 'ARHMM_SL':
        hyperparams['ar_kwargs'] = ar_kwargs
    
    return hyperparams

def arhmm_bayesian_fit(arhmm, data_train, data_test, opt, i_fold):
    
    # Add test data to ARHMM
    for data in data_train:
        # Add data per trial
        arhmm.add_data(data)
        
    #Create data structures to contain gibb samples
    nGibbs = opt['nGibbs']
    nTrials = len(data_train)
    K = arhmm.num_states; D_obs = arhmm.D;
    
    stateseq_smpls = [[] for i in range(nTrials)]
    AB_smpls = np.zeros((nGibbs,K,D_obs,D_obs+1))
    sqrt_sigmas_smpls = np.zeros((nGibbs,K,D_obs,D_obs))
    trans_matrix_smpls = np.zeros((nGibbs,K,K))
    GibbsLLs = np.zeros((nGibbs))
    
    # Loop over samples
    for iSample in tqdm(range(nGibbs)):
        # Sample Model
        arhmm.resample_model()

        #keep track of model log_likelihood's as a check for "convergence"
        GibbsLLs[iSample] = arhmm.log_likelihood()
        
        # Append each Gibbs sample for each trial
        for iTrial in range(len(arhmm.states_list)):
            stateseq_smpls[iTrial].append(arhmm.states_list[iTrial].stateseq.copy())

        # Append the ARHMM matrix A and transition matrix for this sample
        for state in range(K):
            AB_smpls[iSample,state] = arhmm.obs_distns[state].A.copy()
            sqrt_sigmas_smpls[iSample,state] = np.linalg.cholesky(arhmm.obs_distns[state].sigma)
        trans_matrix_smpls[iSample] = arhmm.trans_distn.trans_matrix.copy()
    
    # Calculate the mean A, B, and transition matrix for all
    burn = opt['burn_fraction']
    ABs_mean = np.mean(AB_smpls[int(burn*nGibbs):],axis=0)
    As = ABs_mean[:,:,:D_obs]; Bs = ABs_mean[:,:,D_obs]
    sqrt_Sigmas = np.mean(sqrt_sigmas_smpls[int(burn*nGibbs):],axis=0)
    obs = {'ABs': ABs_mean, 'As': As,'Bs': Bs, 'sqrt_Sigmas': sqrt_Sigmas}

    log_mean_transition_matrix = np.log(np.mean(trans_matrix_smpls[int(burn*nGibbs):,:,:],axis=0))
    trans = {'log_Ps': log_mean_transition_matrix}
    init = {'P0': arhmm.init_state_distn.pi_0}
    
    param_dict = {}
    param_dict['transitions'] = trans
    param_dict['observations'] = obs
    param_dict['init_state_distn'] = init
    
    #llhood of heldout
    ll_heldout = arhmm.log_likelihood(data=data_test)
    state_usage = arhmm.state_usages
    
    #Lists to contain import stuffff
    trMAPs = []
    trPosteriors = []
    trMasks = []
    
    #Plot convergence here
    SaveDir, fname_sffx = util.make_sub_dir(K, opt, i_fold)
    plots_YAA.plot_model_convergence(stateseq_smpls, AB_smpls, trans_matrix_smpls, GibbsLLs, sorted(arhmm.used_states), SaveDir, fname='-'.join(('Model_convergence',fname_sffx))+'.pdf')
    
    #All of the data has been used to fit the model
    #All of the data is contained with the ARHMM object already
    if i_fold == -1:
        #Calculate the MAP estimate
        for iTrial in range(nTrials):
            # Take the gibbs samples after the burn fraction to construct MAP
            z_smpls = np.array(stateseq_smpls[iTrial][int(burn*nGibbs):])

            state_probs_trial = []
            for state in range(K):
                state_occurances = np.isin(z_smpls,state)
                state_probs_trial.append(np.sum(state_occurances,axis=0)/z_smpls.shape[0])

            #Save the maximum posterior probability for each time step
            pprob = np.vstack((np.zeros((1,K)),np.array(state_probs_trial).T))
            trPosteriors.append(pprob)
            mask = np.max(pprob,axis=1) < opt['MAP_threshold']
            trMasks.append(mask)
            
            #Use the maximum posterior probability to determine a robust MAP State sequence
            MAP = np.hstack(([-1],np.ndarray.flatten(st.mode(z_smpls)[0])))
                        
            #Add MAP to list           
            trMAPs.append(MAP)
            ll_heldout
    #Else this is a fold of the x-validation
    else:
        #Get the state sequences and state marginal distributions of the heldout data
        for data in data_test:           
            #Get state marginals
            state_marginals = arhmm.heldout_state_marginals(data)
            trPosteriors.append(state_marginals)
            
            #Create mask
            mask = np.max(state_marginals,axis=1) < opt['MAP_threshold']
            trMasks.append(mask)
            
            #Get the state sequence with the max probability
            stateseq = np.argmax(state_marginals,axis=1)
            trMAPs.append(stateseq)
            
    return trMAPs, trPosteriors, trMasks, state_usage, ll_heldout, param_dict, GibbsLLs

def map_seq_n_usage(arhmm, data_test, opt, inputs=None):
    """
    Compute the local MAP state (arg-max of marginal state probabilities at each time step)
    and overall state usages.
    thresh: if marginal probability of MAP state is below threshold, replace with np.nan
    (or rather output a mask array with nan's in those time steps)

    Also output average state usages and the marginal state probabilities
    """
    T = 0; ll_heldout = 0
    state_usage = np.zeros(arhmm.K)
    trMAPs = []
    trPosteriors = []
    trMasks = []
    
    #Loop over data to obtain MAP sequence for each trial
    for index, data in enumerate(data_test):
        #Get state probabilities and log-likelihood 
        if opt['inputdriven']:
            inputdata = inputs[index]
            Ez, _, ll = arhmm.expected_states(data,input=inputdata)
        else:
            Ez, _, ll = arhmm.expected_states(data)
        
        #Update number of data points, state usage, and llood of data 
        T += Ez.shape[0]
        state_usage += Ez.sum(axis=0)
        ll_heldout += ll
        
        #maximum a posteriori probability estimate of states
        map_seq = np.argmax(Ez,axis=1)
        max_prob = Ez[np.r_[0:Ez.shape[0]],map_seq]
        
        #Save sequences
        trMAPs.append(map_seq)
        trPosteriors.append(Ez)
        trMasks.append(max_prob < opt['MAP_threshold'])
    
    #Normalize
    state_usage /= T
    
    #Get parameters from ARHMM object
    param_dict = util.params_to_dict(arhmm.params, HMM_INPUTS = opt['inputdriven'], ROBUST = opt['robust'])
    
    return trMAPs, trPosteriors, trMasks, state_usage, ll_heldout, param_dict
    
def fit_arhmm_get_llhood(data_list, trsum, K, opt, train_inds=None, test_inds=None, i_fold=-1):
    #Go!
    startTime = time.time()
    
    #Separate the data into a training and test set based on the indices given
    if train_inds is not None and test_inds is not None:
        data_train = [data_list[ii] for ii in train_inds]    
        data_test = [data_list[ii] for ii in test_inds]
        trsum_test = trsum.iloc[test_inds]
    else: 
        #fit model on all data
        data_train = data_list
        data_test = data_list
        trsum_test = trsum
    
    #adding 10 so i_fold == -1 case doesn't give error
    np.random.seed(10+i_fold)
    
    # set hyperparameters
    D_obs, M, Mobs, observation_type, ar_kwargs, transition_type, trans_kwargs = set_arhmm_hyperparams(opt,K)

    ##===== Create the ARHMM object either from Scott's package =====##
    if opt['model_type'] == 'ARHMM_SL':
        arhmm = ssm.HMM(K, D_obs, M=M,
                  observations=observation_type, observation_kwargs=ar_kwargs,
                  transitions=transition_type, transition_kwargs=trans_kwargs) 
        if opt['inputdriven']:
            #Separate inputs from the data_list into training and test sets
            raise Exception('TODO: Separate inputs from the data_list into training and test sets')
            
        else:
            inputs_train = None
            inputs_test = None
            
        ##===== Fit on training data =====##
        model_convergence = arhmm.fit(data_train, inputs=inputs_train, method="em", num_em_iters=opt['EM_iters'], tolerance=opt['EM_tolerance'])
        
        #Get MAP sequences for heldout data (or all of the data if this isn't part of the xval) 
        trMAPs, trPosteriors, trMasks, state_usage, ll_heldout_old, param_dict = map_seq_n_usage(arhmm, data_test, opt, inputs_test)
        
        #Calculate loglikehood of the test and training data
        ll_heldout = arhmm.log_likelihood(data_test)
        ll_training = arhmm.log_likelihood(data_train, inputs=inputs_train)
        
    ##===== Or from Matt Johnson's packages =====##
    else:
        #Sticky or Standard ARHMM
        if opt['sticky']:
            # Create AR-HDP-HMM Object
            arhmm = pyhmm.ARWeakLimitStickyHDPHMM(
                init_state_distn='uniform',
                init_emission_distn=init_distn,
                obs_distns=ar_kwargs,
                alpha=1.0, kappa=opt['kappa'], gamma=3.0)
        else:
            #Vanilla ARHMM
            arhmm = pyhmm.ARHMM(
                alpha=4.,
                init_state_distn='uniform',
                obs_distns=ar_kwargs)
        
        ##===== Fit on training data =====##
        trMAPs, trPosteriors, trMasks, state_usage, ll_heldout, param_dict, model_convergence = \
                arhmm_bayesian_fit(arhmm, data_train, data_test, opt, i_fold)
        
        #Calculate loglikehood of training data
        ll_training = arhmm.log_likelihood()

    #Sort based on state-usage
    trMAPs, trPosteriors, state_usage, state_perm = util.sort_states_by_usage(state_usage, trMAPs, trPosteriors)

    ##===== Calculate Log-likelihood =====##
    #Count total number of time steps in data
    tTest = sum(map(len, data_test))
    ll_heldout_perstep = ll_heldout/tTest
    
    #For Training
    tTrain = sum(map(len, data_train))
    ll_training_perstep = ll_training/tTrain
    llhood_tuple = (ll_heldout,ll_heldout_perstep,ll_training,ll_training_perstep)
    
    ##===== Save & Plot =====##
    #Create subdirectory under base directory for kfold
    SaveDir, fname_sffx = util.make_sub_dir(K, opt, i_fold)
    
    #Convert hyperparameters into dictionary for save process
    hyperparams = make_hyperparams_dic(opt, K, M, trans_kwargs, ar_kwargs)
    
    RunTime = time.perf_counter() - startTime 
    #Plot model parameters
    plots_YAA.save_plot_parameters(SaveDir, fname_sffx, llhood_tuple, state_usage, hyperparams, param_dict, state_perm, model_convergence, RunTime) 
       
    #if this is fit on all data (which i_fold==-1 signifies) plot and save MAP seqs (and state-posteriors)
#     if i_fold == -1:   
#         plots_YAA.save_plot_MAPseqs(SaveDir, fname_sffx, trsum, trMAPs, trPosteriors, trMasks, state_usage, opt, K, state_perm) 
    
    
    return ll_training_perstep, ll_heldout_perstep, K
        
##===== ===== =====##
##===== Start =====##
if __name__ == "__main__":
    #GO! 
    startTime = time.time()                              

    #Convert arguments into dictionary; opt <-> options
    opt = args.__dict__
     
    #Create base folder for saved results    
    SaveDirRoot = util.make_base_dir(opt['model_type'],opt['data_type'],opt['mID'])    
                
    #Save script options in JSON file
    opt['SaveDirRoot'] = SaveDirRoot
    opt['json_dir'] = SaveDirRoot
    js_fname = 'ARHMM_hyperparameters.json'
    if opt['save']:
        with open(os.path.join(SaveDirRoot, js_fname), 'w') as jsfile:
            json.dump(opt, jsfile, indent=4)
            
    ##====== ======================== ======##
    ##====== Read in Observation Data ======##
    data_list, trsum, angle_list = util.read_data(opt['mID'],opt['condition'],opt['data_type'])
    
    # Number of obserations per time step
    D_obs = data_list[0].shape[1]
    opt.update(D_obs = D_obs)
    # Total Trials
    nTrials = len(trsum)
    
    #Save which trials are being used to fit the ARHMM 
    if opt['save']:
        trsum.to_csv(os.path.join(SaveDirRoot,'inputted_trials.txt'),header=False,index=False,sep='\t',float_format='%.4f')
        
    ##===== ==================== =====##
    ##===== Perform X-validation =====##
    k_fold = StratifiedKFold(n_splits=opt['kXval']) 
                     
    #Stratify data per mice and per condition for kfolds
    include = ['{}_C{}'.format(i,j) for i,j in zip(list(trsum['mID']),list(trsum['cond']))]

    # Creates parallel processes
    pool = multiprocessing.Pool(processes=opt['max_processes'])
    
    #Preallocate matrix for cross-validation llhood values
    Ks = np.arange(opt['Kmin'],opt['Kmax']+1,10)
    ll_heldout = np.zeros((len(Ks),opt['kXval']+1))
    ll_training = np.zeros((len(Ks),opt['kXval']+1))
    model_fullfit = []
    process_outputs = []
    
    #Loop over number of HMM states
    for index, K in enumerate(np.arange(opt['Kmin'],opt['Kmax']+1,20)):
        #Fit the model to all of the data, and then for each kfold of x-validation
        model_fullfit.append(pool.apply_async(fit_arhmm_get_llhood, args=(data_list,trsum,K,opt)))
        
        #Loop over kfolds
        kfold_outputs = []
        for iK, (train_indices, test_indices) in enumerate(k_fold.split(data_list, include)):
            kfold_outputs.append(pool.apply_async(fit_arhmm_get_llhood, args= \
                                (data_list, trsum, K, opt, train_indices, test_indices, iK)))
        process_outputs.append(kfold_outputs)
    
    ##===== =========== =====##
    ##===== Get results =====##
    #Extract log_likelihood results from parallel kfold processing
    for index, results in enumerate(process_outputs):
        ll_training[index,:-1] = np.array([iFold.get()[0] for iFold in results])
        ll_heldout[index,:-1] = np.array([iFold.get()[1] for iFold in results])
        Ks[index] = results[0].get()[2]
        time
        
    #For full fit
    Ks_ff = Ks.copy()
    for index, results in enumerate(model_fullfit):
        ll_training[index,-1] = results.get()[0]
        ll_heldout[index,-1] = results.get()[1]
        Ks_ff = results.get()[2]

    #Close Parallel pool
    pool.close()
    
    #Total Run Time
    RunTime = time.perf_counter() - startTime   
    opt.update(RunTime = RunTime)
    hrs=int(RunTime//3600); mins=int((RunTime%3600)//60); secs=int(RunTime - hrs*3600 - mins*60)
    print('\tTotal run time = {:02d}:{:02d}:{:02d} for {} total trials and {} K\'s\n'.format(hrs,mins,secs,nTrials,opt['Kmax']+1-opt['Kmin']))

    # Save summary data of all x-validation results 
    plots_YAA.save_plot_xval_lls(ll_training, ll_heldout, Ks,opt)
