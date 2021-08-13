#Misc
import time, os, sys, pdb, argparse
from glob import glob
from fnmatch import fnmatch

#Base
import numpy as np
import numpy.ma as ma
import pandas as pd
import scipy.io as sio
import scipy.stats as st
import multiprocessing

#Save
import json
import scipy.io as sio
import h5py
import io_dict_to_hdf5 as ioh5

#Plot
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
from matplotlib.backends.backend_pdf import PdfPages

#Model
import ssm
from sklearn.model_selection import StratifiedKFold

#User
import util
import plotting as usrplt
behav_dict = {-1:'ambiguous', 0:'rest',1:'running'}

#Directories
RootDataDir = './data'
ResultsDir = './results'


##===== ============================ =====##
##===== Parse Command Line Arguments =====##
parser = argparse.ArgumentParser(description='HMM for jumping data')

parser.add_argument('--save',type=bool, default=1,
                    help='Save Results?')

##===== Data Options =====##
parser.add_argument('--mID',type=str, default='all_mice',
                    help='mouse to fit model to')

##===== Model Type =====##
parser.add_argument('--model_type', type=str, default='ARHMM',
                    help='ARHMM or SLDS')
parser.add_argument('--transitions', type=str, default='recurrent',
                    help='standard or recurrent or sticky or inputdriven')
parser.add_argument('--observations', type=str, default='autoregressive',
                    help='autoregressive or robust_autoregressive or diagonal_ar or diagonal_robust_ar')
parser.add_argument('--inputdriven', type=bool, default=0,
                    help='HMM transitions dependent on some input in addition to previous HMM state')

##===== Model Parameters =====##
parser.add_argument('--kappa', type=float, default=1e5,
                    help='sticky arhmm kappa')
parser.add_argument('--AR_lags', type=str, default=1,
                    help='Autoregressive lags')
parser.add_argument('--MAP_threshold', type=float, default=0.75,
                    help='MAP threshold')

##===== Run Options =====##
parser.add_argument('--Kmin', type=int, default=4,
                    help='minimum number of HMM states')
parser.add_argument('--Kmax', type=int, default=24,
                    help='maximum number of HMM states')
parser.add_argument('--kXval', type=int, default=5,
                    help='number of kfold')
parser.add_argument('--EM_tolerance', type=float, default=1e-6,
                    help='SSM EM algorithm tolerance')
parser.add_argument('--EM_iters', type=int, default=200,
                    help='EM Iterations')
parser.add_argument('--max_processes', type=int, default=15,
                    help='max # of parallel processes to run')
args = parser.parse_args()


def set_arhmm_hyperparams(opt,K):
    
    Mobs = 0
    
    #Autoregressive keyword arguments
    ar_kwargs = dict(
            # l2_penalty_A= args_dic['l2_penalty_A'],
            # l2_penalty_b= args_dic['l2_penalty_b'],
            # l2_penalty_V= args_dic['l2_penalty_V'],
            lags = opt['AR_lags']
            )
    
    #HMM Transition parameters 
    if opt['transitions'] == 'sticky':
        # alpha= args_dic['alpha'],
        trans_kwargs['kappa'] = opt['kappa']
    else:
        trans_kwargs = {}
        
    return M, ar_kwargs, trans_kwargs

def get_state_sequence(hmm, data_test, opt, inputs=None):
    """
    Compute the local MAP state (arg-max of marginal state probabilities at each time step)
    and overall state usages.
    thresh: if marginal probability of MAP state is below threshold, replace with np.nan
    (or rather output a mask array with nan's in those time steps)

    Also output average state usages and the marginal state probabilities
    """
    T = 0; ll_heldout = 0
    state_usage = np.zeros(hmm.K)
    trMAPs = []
    trPosteriors = []
    trMasks = []
    
    #Loop over data to obtain MAP sequence for each trial
    for index, data in enumerate(data_test):
        #Get state probabilities and log-likelihood 
        if opt['transitions'] == 'inputdriven':
            inputdata = inputs[index]
            Ez, _, ll = hmm.expected_states(data,input=inputs)
        else:
            Ez, _, ll = hmm.expected_states(data)
        
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
        trMasks.append(max_prob > opt['MAP_threshold'])
    
    #Normalize
    state_usage /= T
    
    #Get parameters from ARHMM object
    param_dict = util.params_to_dict(hmm.params, opt)
    
    return trMAPs, trPosteriors, trMasks, state_usage, ll_heldout, param_dict
    
def fit_ssm_get_llhood(data_list, K, opt, train_inds=None, test_inds=None, i_fold=-1):
    #Go!
    startTime = time.time()
    
    nTrials = len(data_list)
    #Separate the data into a training and test set based on the indices given
    if train_inds is not None and test_inds is not None:
        data_train = [data_list[ii] for ii in train_inds]    
        data_test = [data_list[ii] for ii in test_inds]
    else: 
        #fit model on all data
        data_train = data_list
        data_test = data_list

    #adding 10 so i_fold == -1 case doesn't give error
    np.random.seed(10+i_fold)
    
    #Autoregressive keyword arguments
    ar_kwargs = dict(
            # l2_penalty_A= args_dic['l2_penalty_A'],
            # l2_penalty_b= args_dic['l2_penalty_b'],
            # l2_penalty_V= args_dic['l2_penalty_V'],
            lags = opt['AR_lags']
            )
    
    #HMM Transition parameters 
    if opt['transitions'] == 'sticky':
        # alpha= args_dic['alpha'],
        trans_kwargs['kappa'] = opt['kappa']
    else:
        trans_kwargs = {}
        
    #Not implemented yet
    if opt['transitions'] == 'inputdriven':
        #Separate inputs from the data_list into training and test sets
        raise Exception('TODO: Separate inputs from the data_list into training and test sets')
    else:
        inputs_train = None
        inputs_test = None
        M = 0
            
    #Initialize Hidden Markov Model with
    arhmm = ssm.HMM(K, opt['dObs'], M=M,
                  observations=opt['observations'], observation_kwargs=ar_kwargs,
                  transitions=opt['transitions'], transition_kwargs=trans_kwargs) 

    ##===== Fit on training data =====##
    model_convergence = arhmm.fit(data_train, inputs=inputs_train, method="em", num_iters=opt['EM_iters'], tolerance=opt['EM_tolerance'])
    
    #Get MAP sequences for heldout data (or all of the data if this isn't part of the xval) 
    trMAPs, trPosteriors, trMasks, state_usage, ll_heldout2, params_dict = get_state_sequence(arhmm, data_test, opt)

    #Calculate loglikehood of the test and training data
    ll_heldout = arhmm.log_likelihood(data_test)
    ll_training = arhmm.log_likelihood(data_train)
   
    #Sort based on state-usage
#     trMAPs, trPosteriors, state_usage, state_perm = util.sort_states_by_usage(state_usage, trMAPs, trPosteriors)
    
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
    
    #Stop time
    RunTime = time.perf_counter() - startTime 

    ## Save log-likelihood per kfold fit, as well as fit model parameters
    ioh5.save(os.path.join(SaveDir, 'fit_parameters-{}.h5'.format(fname_sffx)),
                    {'ll_heldout':llhood_tuple[0], 'll_heldout_perstep':llhood_tuple[1],'tTest': tTest, 
                    'll_training':llhood_tuple[2], 'll_training_perstep':llhood_tuple[3],'tTrain': tTrain, 
                    'state_usage':state_usage, 'arhmm_params' : params_dict,'hyperparams': opt, 
                    'model_convergence': model_convergence, 'RunTime': RunTime})  
    
    ##===== Save and plot for full fit =====##
    if i_fold == -1:
        
        ## Save state sequences for full fit
        ioh5.save(os.path.join(SaveDir, 'MAP_seqs-{}.h5'.format(fname_sffx)), 
          {'trMAPs':trMAPs, 'trPosteriors':trPosteriors,'trMasks':trMasks, 
           'arhmm_params' : params_dict,'state_usage':state_usage, 
           'hyperparams' : opt})
        
        ## Calculate & plot state duration and state usage 
        state_duration_list, state_startend_list, state_usage = util.get_state_durations(trMAPs, trMasks, K)
        usrplt.plot_state_durations2(state_duration_list,state_usage, K,
                                    SAVEFIG=True,PlotDir=SaveDir,fname='state-durations_{}.pdf'.format(fname_sffx))
        
        #Plot dynamics of latent states
        usrplt.plot_dynamics_2d(arhmm,SAVEFIG=True,PlotDir=SaveDir,fname='AR-streamplots_{}.pdf'.format(fname_sffx))
        
        ## Plot the actual AR matrices, with their corresponding fixed point
        usrplt.plot_AR_matrices(arhmm,SAVEFIG=True,PlotDir=SaveDir,fname='AR-matrices_{}.pdf'.format(fname_sffx))
        
        ## Plot example trajectories of actual trajectories for each state
        usrplt.plot_example_trajectories(state_duration_list,state_startend_list,data_list, arhmm,
                                        SAVEFIG=True,PlotDir=SaveDir,fname='state-trajectories_data_{}.pdf'.format(fname_sffx))
        
        ## Plot example trajectories simulated from the model for each state
        usrplt.plot_example_trajectories(state_duration_list,state_startend_list,data_list, arhmm, simulated=True,
                                        SAVEFIG=True,PlotDir=SaveDir,fname='state-trajectories_simulated_{}.pdf'.format(fname_sffx))

    
    return ll_training_perstep, ll_heldout_perstep, K
        
##===== ===== =====##
##===== Start =====##
if __name__ == "__main__":
    #GO! 
    startTime = time.time()                              

    #Convert arguments into dictionary; opt <-> options
    opt = args.__dict__
     
    #Create base folder for saved results    
    SaveDirRoot = util.make_base_dir(opt['model_type'],opt['mID'])    
                
    #Save script options in JSON file
    opt['SaveDirRoot'] = SaveDirRoot
    if opt['save']:
        with open(os.path.join(SaveDirRoot, 'ARHMM_hyperparameters.json'), 'w') as jsfile:
            json.dump(opt, jsfile, indent=4)
            
    ##====== ============ ======##
    ##====== Read in Data ======##
    data_df = pd.read_hdf('./data/jumping_data_102220.h5')
    nTrials = len(data_df)
    
    #DLC tracking confidence threshold at which to mask out data
    confidence_threshold = 0.8

    #Loop over trials and reformat data for ARHMM
    data_list = []; mask_list = []
    for iTrial in range(nTrials):
        #Get coordinates of Take-Off platform
        xc = np.nanmean(data_df.loc[iTrial]['Side TakeFL x'])
        yc = np.nanmean(data_df.loc[iTrial]['Side TakeFL y'])

        xy_list = []; ll_list = []
        for ii, ptstr in enumerate(['Nose','LEye','LEar']):
            x = data_df.loc[iTrial]['Side {} x'.format(ptstr)]
            y = data_df.loc[iTrial]['Side {} y'.format(ptstr)]
            llhood = data_df.loc[iTrial]['Side {} likelihood'.format(ptstr)]

            #Coordinates relative to take-off platform
            xy_list.append((x-xc,y-yc))

            #Create mask for points that have a confidence lower than the given threshold
            mask = llhood > confidence_threshold
            ll_list.append((mask,mask))

        tmp = np.vstack(xy_list).T; data_list.append(tmp[2:-2,:])
        tmp = np.vstack(ll_list).T; mask_list.append(tmp[2:-2,:])

    #Get number of time points and components per experiment
    nT, dObs = data_list[0].shape
    nComponents = dObs; opt['dObs'] = dObs

    ##===== ==================== =====##
    ##===== Perform X-validation =====##
    k_fold = StratifiedKFold(n_splits=opt['kXval'])
    #Stratify data per mice and per condition for kfolds
    include = ['{}_D{}'.format(i,j) for i,j in zip(list(data_df['subject']),list(data_df['distance']))]
                     
    # Creates parallel processes
    pool = multiprocessing.Pool(processes=opt['max_processes'])
    
    #Preallocate matrix for cross-validation llhood values
    Ks = np.arange(opt['Kmin'],opt['Kmax']+1,2)
    ll_heldout = np.zeros((len(Ks),opt['kXval']+1))
    ll_training = np.zeros((len(Ks),opt['kXval']+1))
    model_fullfit = []
    process_outputs = []

    #Loop over number of HMM states
    for index, K in enumerate(np.arange(opt['Kmin'],opt['Kmax']+1,2)):
        print('{} states'.format(K))
        #Fit the model to all of the data, and then for each kfold of x-validation
        model_fullfit.append(pool.apply_async(fit_ssm_get_llhood, args=(data_list,K,opt)))
#         ll_training_perstep, ll_heldout_perstep, K = fit_ssm_get_llhood(data_list,K,opt)

        #Loop over kfolds
        kfold_outputs = []
        for iK, (train_indices, test_indices) in enumerate(k_fold.split(data_list,include)):
            kfold_outputs.append(pool.apply_async(fit_ssm_get_llhood, args= \
                                (data_list, K, opt, train_indices, test_indices, iK)))
        process_outputs.append(kfold_outputs)
    
    ##===== =========== =====##
    ##===== Get results =====##
    #Extract log_likelihood results from parallel kfold processing
    for index, results in enumerate(process_outputs):
        ll_training[index,:-1] = np.array([iFold.get()[0] for iFold in results])
        ll_heldout[index,:-1] = np.array([iFold.get()[1] for iFold in results])
        Ks[index] = results[0].get()[2]
       
    #For full fit
    Ks_ff = Ks.copy()
    for index, results in enumerate(model_fullfit):
        ll_training[index,-1] = results.get()[0]
        ll_heldout[index,-1] = results.get()[1]
        Ks_ff = results.get()[2]

    #Close Parallel pool
    pool.close()
#     pdb.set_trace()
    #Total Run Time
    RunTime = time.perf_counter() - startTime   
    opt.update(RunTime = RunTime)
    hrs=int(RunTime//3600); mins=int((RunTime%3600)//60); secs=int(RunTime - hrs*3600 - mins*60)
    print('\tTotal run time = {:02d}:{:02d}:{:02d} for {} K\'s\n'.format(hrs,mins,secs,opt['Kmax']+1-opt['Kmin']))

    # Save summary data of all x-validation results 
    usrplt.plot_xval_lls_vs_K(ll_training, ll_heldout, Ks, opt, SAVEFIG=True)
    xval_fname = '{}_lls_vs_K_{}to{}'.format(opt['mID'],opt['Kmin'],opt['Kmax'])
    ioh5.save(os.path.join(opt['SaveDirRoot'], xval_fname+'.h5'), {'ll_heldout':ll_heldout, 'll_training':ll_training,'Ks': Ks,'RunTime':opt['RunTime'], 'kXval':opt['kXval'], 'Kmin':opt['Kmin'], 'Kmax':opt['Kmax']})
