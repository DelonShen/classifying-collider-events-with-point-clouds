import ROOT
from ROOT import TLorentzVector

from tqdm import trange
import pickle
from math import isnan
import numpy as np
from statistics import mode
import tensorflow as tf
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import matplotlib

import sys
sys.path.insert(1, '../')

from Architectures import *

import random

np.random.seed(42)
tf.random.set_seed(42)

identity = lambda *args: args
#MASS FROM PDG
#SIGMA FROM https://cds.cern.ch/record/2715303/files/TOPQ-2018-18-002.pdf
m_t = 172.76 #GeV
sigma_t = 10.7 #GeV
m_W =  80.379 #GeV
sigma_W = 5.9 #GeV

from itertools import combinations, accumulate, permutations

def create_ragged_tensor(events):
    rnd_idx = random.randint(0,len(events)-1) 
    chk = np.array(events[rnd_idx])
    values = [jet for event in events for jet in event]
    length = [len(event) for event in events]
    oup = tf.RaggedTensor.from_row_lengths(values = values, row_lengths=length, validate=False)
    assert((chk - oup.numpy()[rnd_idx] < 1e-4).all())
    return oup


def load_data_from_file(filename):
  #load in data
  inpfile = open(filename, 'rb')
  events, events_oup, events_tag  = pickle.load(inpfile)
  inpfile.close()
  
  events = np.array(events)
  events_oup = np.array(events_oup)
  events_tag = np.array(events_tag)

  return events, events_oup, events_tag
  
def train_test_split(length, split=0.7):
  indices = np.random.permutation(length)
  split_len = int(length*split)
  return indices[:split_len], indices[split_len:]

# may put this function in another utility file
def suppress_warnings():
    # Filter tensorflow version warnings
    import os
    # https://stackoverflow.com/questions/40426502/is-there-a-way-to-suppress-the-messages-tensorflow-prints/40426709
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

    import warnings
    # https://stackoverflow.com/questions/15777951/how-to-suppress-pandas-future-warning
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=Warning)

    tf.get_logger().setLevel('INFO')
    tf.autograph.set_verbosity(0)
    import logging
    tf.get_logger().setLevel(logging.ERROR)


  
def class_weight_invariant(y_train):
    n_true = 0
    n_false = 0
    for event in y_train:
        n_true  += event[0]
        n_false += event[1]
    weight_for_0 = (1 / n_true)  * (len(y_train) / 2.0)
    weight_for_1 = (1 / n_false) * (len(y_train) / 2.0)

    return weight_for_0, weight_for_1

def class_weight(y_train): #we assume two classes
  #Define Class Rebalancing Constant
  n_true = 0
  n_false = 0
  for event in y_train:
      for target in event:
          n_true  += target[0]
          n_false += target[1]
  weight_for_0 = (1 / n_true)  * (len(y_train) / 2.0)
  weight_for_1 = (1 / n_false) * (len(y_train) / 2.0)

  return weight_for_0, weight_for_1
  
def get_DeltaR(p4_1, p4_2):
    return p4_1.DeltaR(p4_2)**2

def gen_dataset_high_level(inp, oup, events_tag, systematics_test=False):
    #extract input to bdt/nn from table 4 of paper draft
    X = []
    y = []

    for event_idx in range(len(inp)):
        curr_event = []
        curr_oup = []

        curr_event_tag = events_tag[event_idx][0]
        curr_oup = [curr_event_tag==0, curr_event_tag==1]
        curr_oup = list(map(int, curr_oup))

        temp_jets = inp[event_idx].copy()
        n_jets = len(temp_jets)


        curr_event.append(sum([temp_jets[jet_idx][0].Perp() for jet_idx in range(n_jets)])) #Scalar sum of all jets pT

        jet_idxs = [i for i in range(n_jets)]
        tau_idxs = [i for i in range(n_jets) if temp_jets[i][2]==1]
        assert(len(tau_idxs)==2)

        #Best W-candidate dijet invariant mass
        #AND smallest DeltaR
        multijet_idxs = combinations(jet_idxs, 2)
        curr_best_W_mass = -1.0
        curr_best_W_mass_diff = 1.1e10

        curr_smallest_DR = 1.1e10
        for jet_idxs in multijet_idxs:
            jetP4s = [temp_jets[jet_idx][0] for jet_idx in jet_idxs]

            dijetDR = jetP4s[0].DeltaR(jetP4s[1])
            if(dijetDR < curr_smallest_DR):
                curr_smallest_DR = dijetDR


            multijetP4 = TLorentzVector()
            for jetP4 in jetP4s:
                multijetP4 += jetP4
            multijet_mass = multijetP4.M()
#            print(multijet_mass, m_W)
            mass_diff = abs(multijet_mass - m_W)
            if(mass_diff < curr_best_W_mass_diff):
                curr_best_W_mass_diff = mass_diff
                curr_best_W_mass = multijet_mass
        curr_event.append(curr_best_W_mass)
        curr_event.append(curr_smallest_DR)

        #Best t-candidate dijet invariant mass
        multijet_idxs = combinations(jet_idxs, 3)
        curr_best_t_mass = -1.0
        curr_best_t_mass_diff = 1.1e10
        for jet_idxs in multijet_idxs:
            jetP4s = [temp_jets[jet_idx][0] for jet_idx in jet_idxs]
            multijetP4 = TLorentzVector()
            for jetP4 in jetP4s:
                multijetP4 += jetP4
            multijet_mass = multijetP4.M()
#            print(multijet_mass, m_t)
            mass_diff = abs(multijet_mass - m_t)
            if(mass_diff < curr_best_t_mass_diff):
                curr_best_t_mass_diff = mass_diff
                curr_best_t_mass = multijet_mass
        curr_event.append(curr_best_t_mass)

        #Delta R tau tau and |Delta Eta(tau tau)| and pT(tautau)
        tau_jet_P4 = [temp_jets[jet_idx][0] for jet_idx in tau_idxs]

        DR_tautau = tau_jet_P4[0].DeltaR(tau_jet_P4[1])
        curr_event.append(DR_tautau)

        DEta_tautau = abs(tau_jet_P4[0].Eta() - tau_jet_P4[1].Eta())
        curr_event.append(DEta_tautau)

        pT_tautau = (tau_jet_P4[0] + tau_jet_P4[1]).Perp()
        curr_event.append(pT_tautau)

        #MET
        MET = events_tag[event_idx][2]

        curr_event.append(MET)





        X.append(curr_event)
        y.append(curr_oup)

    return X,y


def gen_multijet_to_inv_dataset(inp, oup, events_tag, multijet_n, dR_keep = True, pad=True, pad_n=15, systematics_test=False, MET_jet=False):
    X = []
    y = []

    max_rag = -1
    for event_idx in range(len(inp)):
        curr_multijets = []
        curr_oup = []

        curr_event_tag = events_tag[event_idx][0]
        curr_oup = [curr_event_tag==0, curr_event_tag==1]
        curr_oup = list(map(int, curr_oup))

        temp_jets = [jet for jet in inp[event_idx]]
        temp_jets_pt = [jet[0].Perp() for jet in inp[event_idx]]
        sort_idxs = np.argsort(temp_jets_pt)[::-1]
        temp_jets = np.array(temp_jets)[sort_idxs]

        n_jets = len(temp_jets)

        jetP4_tot = TLorentzVector()
        jetPT_tot = 0
        jetM_tot = 0
        vectPT = [0,0]

        for jet_idx in range(n_jets):
            temp_jet_P4 = temp_jets[jet_idx][0]
            jetP4_tot += temp_jet_P4
            jetPT_tot += temp_jet_P4.Perp()
            vectPT[0] += temp_jet_P4.Px()
            vectPT[1] += temp_jet_P4.Py()
            jetM_tot += np.abs(temp_jet_P4.M())
            assert(abs(temp_jets[jet_idx][0].M()- np.abs(temp_jets[jet_idx][0].M()) ) <= 1e-3)

        jet_idxs = [i for i in range(n_jets)]
        multijet_idxs = combinations(jet_idxs, multijet_n)

        for jet_idxs in multijet_idxs:
            jetP4s = [temp_jets[jet_idx][0] for jet_idx in jet_idxs]

            multijetP4 = TLorentzVector()
            for jetP4 in jetP4s:
                multijetP4 += jetP4

            multijetE = multijetP4.E()
            multijetPT = multijetP4.Perp()
            multijety = multijetP4.PseudoRapidity()
            multijetP = multijetP4.Phi()
            multijetM = np.abs(multijetP4.M())
            assert(abs(multijetP4.M() - abs(multijetP4.M())) <= 1e-3)


            multijetDR = 1
            if(multijet_n > 1):
                jet_P4combos = list(combinations(jetP4s, 2))
                jet_P4combos = list(zip(*jet_P4combos))
                jet_P4combos = list(map(get_DeltaR, jet_P4combos[0], jet_P4combos[1]))
                multijetDR = sum(jet_P4combos)



            jet_btags = [temp_jets[jet_idx][1] for jet_idx in jet_idxs]
            jet_tautags = [temp_jets[jet_idx][2] for jet_idx in jet_idxs]

            multijetb = sum(jet_btags)
            multijettau = sum(jet_tautags)

            eps = 1e-4
            tmp_log = np.log(eps+multijetM/jetM_tot)
            if(np.isnan(tmp_log) or np.isinf(tmp_log)):
                print(multijetM, jetM_tot, 1- multijetM/jetM_tot)
                assert(1==0)
            curr_multijet = [np.log(multijetE/jetP4_tot.E()), np.log(eps+multijetM/jetM_tot), multijety, multijetP, np.log(multijetPT/jetPT_tot), multijetb/multijet_n, multijettau/multijet_n]
#            curr_multijet = [np.log(multijetE/jetP4_tot.E()), multijety, multijetP, np.log(multijetPT/jetPT_tot), multijetb/multijet_n, multijettau/multijet_n]
#            curr_multijet = [(multijetE), (multijetM), multijety, multijetP, (multijetPT), multijetb/multijet_n, multijettau/multijet_n]
            if(dR_keep):
                curr_multijet.append(np.log(multijetDR))

            curr_multijets.append(curr_multijet)
        assert(len(curr_multijets)>0)
        X.append(curr_multijets)
        max_rag = max(max_rag, len(curr_multijets))
        y.append(curr_oup)
    if(pad):
        assert(max_rag <= pad_n)
        X = tf.keras.preprocessing.sequence.pad_sequences(X, dtype='float32', padding='post', maxlen=pad_n)
    return X,y

      
 
  

  

def standardize_data(X, y, means=[], stds=[]):
  if(len(means)==0 or len(stds)==0):
    print('std and means not passed, calculating standardization')
    in_features = len(X[0][0])
    #preproccing, normlize each input feature
    features = np.array([jet for event in X for jet in event]).transpose()
    means = np.array([features[i].mean() for i in range(in_features)])
    stds = np.array([features[i].std() for i in range(in_features)])
    
  X = [np.array([list((jet-means)/stds) for jet in event])for event in X]
  y = [np.array(event) for event in (y)]
  
  return X, y, means, stds
  

def calculate_ROC(oup, true, is_tensor=True):
  oup_prime = oup
  true_prime = true

  if(is_tensor):
    ()
    true_prime = true.to_list()
  print('Calculating ROC')
  y_pred = np.array([jet for event in oup_prime for jet in event]).transpose()
  y_true = np.array([jet for event in true_prime for jet in event]).transpose()
#  print(y_true[0], y_pred[0])
  try:
    ftpr = [(roc_curve(y_true[i], y_pred[i]))  for i in range(len(y_pred))]
    roc_auc = [auc(fpr, tpr) for fpr, tpr, _ in ftpr] 
  except:
    ftpr = [(roc_curve(y_true, y_pred))]
    roc_auc = [auc(fpr, tpr) for fpr, tpr, _ in ftpr] 
  return ftpr, roc_auc

def plot_ROC(oup, true, filename, category_names, is_tensor=True,
            name='Top Reconstruction', color=['black', 'black', 'gainsboro'],
            style=['-','--', '--'], use_latex=True):
  
  if(use_latex):
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['axes.unicode_minus'] = False
  else:
    matplotlib.rcParams['text.usetex'] = False

  #plot roc curves
  plt.rcParams['font.family'] = 'serif'
  plt.rcParams['font.size'] = 25
  plt.rcParams['figure.autolayout'] = True
  plt.figure(figsize=(8, 8), dpi=80)

  ftpr, roc_auc = calculate_ROC(oup, true, is_tensor=is_tensor)
  print('roc calculated')
  
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.0])
  if(use_latex):
    annotation_string = r'\textbf{ROC Curve for %s}'%(name)
    annotation_string += '\n'
    annotation_string += r'$t\overline t(H\rightarrow\tau\tau)$, \textsc{Pythia} 8+\textsc{Delphes}'
    annotation_string += '\n'
    annotation_string += r'Anti-Kt with $R=0.4$, $\sqrt s = 14$'
    plt.text(.38,.25, annotation_string)
  
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')     
  for i in range(len(category_names)):
    if(use_latex):
      plt.plot(ftpr[i][0], ftpr[i][1], style[i], color=color[i], label=r'AUC: %.3f, %s'%(roc_auc[i], category_names[i]))
    else:
      plt.plot(ftpr[i][0], ftpr[i][1], style[i], color=color[i], label='AUC: %.3f, %s'%(roc_auc[i], category_names[i]))
  plt.legend(loc='upper left', frameon=False)
  plt.savefig(filename + ".pdf")
  plt.savefig(filename + ".png")


#IMPLEMENTING CUSTOM LOSS TO WEIGH CLASES SINCE TENSORFLOW DOESN'T
#SUPPORT CLASS WEIGHTING FOR RAGGED TENSORS
import tensorflow.keras.backend as K 
from tensorflow.python.keras.losses import _ragged_tensor_apply_loss
from tensorflow.python.ops import math_ops
import functools

def weightedLoss(originalLossFunc, weightsList):
    '''from https://stackoverflow.com/questions/51793737/custom-loss-function-for-u-net-in-keras-using-class-weights-class-weight-not?answertab=votes#tab-top'''
    def lossFunc(true, pred):

        axis = -1 #if channels last 
        #axis=  1 #if channels first


        #argmax returns the index of the element with the greatest value
        #done in the class axis, it returns the class index    
        classSelectors = K.argmax(true, axis=axis) 
            #if your loss is sparse, use only true as classSelectors

        #considering weights are ordered by class, for each class
        #true(1) if the class index is equal to the weight index   
        classSelectors = [K.equal(i, classSelectors) for i in range(len(weightsList))]

        #casting boolean to float for calculations  
        #each tensor in the list contains 1 where ground true class is equal to its index 
        #if you sum all these, you will get a tensor full of ones. 
        classSelectors = [K.cast(x, K.floatx()) for x in classSelectors]

        #for each of the selections above, multiply their respective weight
        weights = [sel * w for sel,w in zip(classSelectors, weightsList)] 

        #sums all the selections
        #result is a tensor with the respective weight for each element in predictions
        weightMultiplier = weights[0]
        for i in range(1, len(weights)):
            weightMultiplier = weightMultiplier + weights[i]


        #make sure your originalLossFunc only collapses the class axis
        #you need the other axes intact to multiply the weights tensor
        loss = originalLossFunc(true,pred) 
        loss = loss * weightMultiplier

        return loss
    return lossFunc


import seaborn as sns
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

def earth_movers_distance(x,y,N=3000):
    print(N/len(x))
    d = cdist(x[:N],y[:N])
    assignment = linear_sum_assignment(d)
    return (d[assignment].sum()/N)


import numpy as np

def compute_kde_estimate(events):
    from scipy.stats import gaussian_kde
    estimator = gaussian_kde(events)
    return estimator
import math
def rotate(p, origin=(0, 0), angle=0):
    #ANGLE IN RADS
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)
    p = p - o
    p = [R@pnt for pnt in p]
    
    return p + o

#FROM https://mail.python.org/pipermail/scipy-user/2011-May/029521.html
#With modificiation to handle singularities 
def KLdivergence(x, y):
    """Compute the Kullback-Leibler divergence between two multivariate samples.

    Parameters
    ----------
    x : 2D array (n,d)
      Samples from distribution P, which typically represents the true
      distribution.
    y : 2D array (m,d)
      Samples from distribution Q, which typically represents the approximate
      distribution.

    Returns
    -------
    out : float
      The estimated Kullback-Leibler divergence D(P||Q).

    References
    ----------
    Pérez-Cruz, F. Kullback-Leibler divergence estimation of
continuous distributions IEEE International Symposium on Information
Theory, 2008.
    """
    from scipy.spatial import cKDTree as KDTree

    # Check the dimensions are consistent
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)

    n,d = x.shape
    m,dy = y.shape

    assert(d == dy)


    # Build a KD tree representation of the samples and find the nearest neighbour
    # of each point in x.
    xtree = KDTree(x)
    ytree = KDTree(y)

    # Get the first two nearest neighbours for x, since the closest one is the
    # sample itself.
    r = xtree.query(x, k=2, eps=.01, p=2)[0][:,1]
    s = ytree.query(x, k=1, eps=.01, p=2)[0]

    lg = np.log(r/s)
    #get rid of singularities
    lg = lg[np.abs(lg) != float('inf')]
    
    # There is a mistake in the paper. In Eq. 14, the right side misses a negative sign
    # on the first term of the right hand side.
    return -lg.sum() * d / n + np.log(m / (n - 1.))

from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
import openTSNE
def compute_tsne_embedded(latent_reps, perplexity=[50, 1998]):
    x = latent_reps
    print(len(x[0]))
    print('computing tsne')
    
    affinities_multiscale_mixture = openTSNE.affinity.Multiscale(
        x,
        perplexities=perplexity,
        metric="euclidean",
        n_jobs=-1,
        random_state=8,
    )
    init = openTSNE.initialization.pca(x, random_state=8)
#     if(len(x[0])==2):
#         return openTSNE.TSNE(n_jobs=-1, verbose=True).fit(affinities=affinities_multiscale_mixture)

    latent_reps_embedded_tsne = openTSNE.TSNE(n_jobs=-1, verbose=True).fit(affinities=affinities_multiscale_mixture,
                                                             initialization=init,)
    

    return latent_reps_embedded_tsne

def compute_tsne(model, cut, X_test, perplexity=[50, 1998]):
    latent_getter = LatentGetter(model.layers[0:3], condensed=True)
    latent_reps = latent_getter.predict(X_test.numpy()[cut])
    return compute_tsne_embedded(latent_reps, perplexity=perplexity)

def emd(events, latent_label):
    ttH_loc = np.array([events[i] for i in range(len(latent_label)) if latent_label[i][1]==1])
    nttH_loc = np.array([events[i] for i in range(len(latent_label)) if latent_label[i][0]==1])
    return earth_movers_distance(ttH_loc, nttH_loc)


def kldiv(events, latent_label):
    ttH_loc = np.array([events[i] for i in range(len(latent_label)) if latent_label[i][1]==1])
    nttH_loc = np.array([events[i] for i in range(len(latent_label)) if latent_label[i][0]==1])
    return KLdivergence(ttH_loc, nttH_loc)

def TEMP_gen_tsne(curr_event, latent_label, text=r'\textbf{Latent Representation} in Pairwise Architecture', rotated=True, standardized=True):
    c_cut = 5
    cmap = sns.cubehelix_palette(start=26/10, light=.97, as_cmap=True)

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 25
    plt.rcParams['figure.autolayout'] = False
    plt.rcParams['text.usetex'] = True
    
    means = [0,0]
    stds = [1,1]
    
    if(standardized):
        means = np.mean(curr_event, axis=0)
        stds = np.std(curr_event, axis=0)
        
    ttH_loc  = np.array([[(curr_event[i][j]-means[j])/stds[j] for j in range(len(curr_event[i]))] 
                         for i in range(len(latent_label)) if latent_label[i][1]==1])
    nttH_loc = np.array([[(curr_event[i][j]-means[j])/stds[j] for j in range(len(curr_event[i]))] 
                         for i in range(len(latent_label)) if latent_label[i][0]==1])

    if(rotated):
        cntr      = np.array([np.mean(curr_event[:,0]), np.mean(curr_event[:, 1])])
        ttH_cntr  = np.array([np.mean(ttH_loc[:,0]), np.mean(ttH_loc[:, 1])])
        nttH_cntr = np.array([np.mean(nttH_loc[:,0]), np.mean(nttH_loc[:, 1])])
        tmp = ttH_cntr - cntr
        tmp = tmp / np.linalg.norm(tmp)
        angl = -np.arctan(tmp[1]/tmp[0])
        ttH_loc = rotate(ttH_loc, origin=cntr, angle=angl)
        nttH_loc = rotate(nttH_loc, origin=cntr, angle=angl)
    
    
    distance = earth_movers_distance(ttH_loc, nttH_loc)
    print('distance', distance)
    

def gen_tsne(curr_event, latent_label, text=r'\textbf{Latent Representation} in Pairwise Architecture', rotated=True, log=False, bnds=False, cmap=sns.cubehelix_palette(start=26/10, light=.97, as_cmap=True), col_aux='#d495f4', EMD=True, NOAXIS=True, standardized=True):
    c_cut = 5

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 25
    plt.rcParams['figure.autolayout'] = False
    plt.rcParams['text.usetex'] = True

  

    means = [0,0]
    stds = [1,1]
    
    if(standardized):
        means = np.mean(curr_event, axis=0)
        stds = np.std(curr_event, axis=0)
        
    ttH_loc  = np.array([[(curr_event[i][j]-means[j])/stds[j] for j in range(len(curr_event[i]))] 
                         for i in range(len(latent_label)) if latent_label[i][1]==1])
    nttH_loc = np.array([[(curr_event[i][j]-means[j])/stds[j] for j in range(len(curr_event[i]))] 
                         for i in range(len(latent_label)) if latent_label[i][0]==1])

    
    if(rotated):
        cntr      = np.array([np.mean(curr_event[:,0]), np.mean(curr_event[:, 1])])
        print(cntr, np.array([np.std(curr_event[:,0]), np.std(curr_event[:, 1])]))
        ttH_cntr  = np.array([np.mean(ttH_loc[:,0]), np.mean(ttH_loc[:, 1])])
        nttH_cntr = np.array([np.mean(nttH_loc[:,0]), np.mean(nttH_loc[:, 1])])
        tmp = ttH_cntr - cntr
        tmp = tmp / np.linalg.norm(tmp)
        angl = -np.arctan(tmp[1]/tmp[0])
        ttH_loc = rotate(ttH_loc, origin=cntr, angle=angl)
        nttH_loc = rotate(nttH_loc, origin=cntr, angle=angl)
        curr_event = rotate(curr_event, origin=cntr, angle=angl)
        
        ttH_cntr  = np.array([np.mean(ttH_loc[:,0]), np.mean(ttH_loc[:, 1])])
        nttH_cntr = np.array([np.mean(nttH_loc[:,0]), np.mean(nttH_loc[:, 1])])
        if(ttH_cntr[0] < nttH_cntr[0]):
            ttH_loc = np.array([[-x0,x1] for x0,x1 in ttH_loc])
            nttH_loc = np.array([[-x0, x1] for x0, x1 in nttH_loc])

    
    distance = earth_movers_distance(ttH_loc, nttH_loc)
    print('earth_movers_distance', distance)
       
    g = sns.jointplot(x=ttH_loc[:,0], y = ttH_loc[:,1], color=cmap(100), space=0, label='ttH jets',
                      cmap=cmap, kind='kde', height=10, fill=True, cut=c_cut,
                     marginal_kws={'linewidth': 0.0, 'alpha':1.0, 'log_scale':log},
                     joint_kws={'log_scale':log})

    linew = 1
    bgrey = sns.dark_palette(col_aux, reverse=True, as_cmap=True)
    COL2 = bgrey(.8)
    sns.kdeplot(x=nttH_loc[:,0], y = nttH_loc[:,1], shade=False, label=r'ttbar jets',cmap=bgrey, linewidths=linew, cut=c_cut, 
                levels=10, ax=g.ax_joint, log_scale=log)

    sns.kdeplot(nttH_loc[:,0], ax=g.ax_marg_x, color=COL2, lw=linew, log_scale=log)
    sns.kdeplot(y=nttH_loc[:,1], ax=g.ax_marg_y, color=COL2, lw=linew, log_scale=log)

    ax = g.ax_joint

    import matplotlib.patches as  mpatches
    import matplotlib.lines as  mlines
    
    handles = [mpatches.Patch(facecolor=cmap(100), label=r'$ttH$ Events'),
               mlines.Line2D([], [], color=COL2, label=r'$t\overline{t}$ Events', lw=linew)]
    legend = ax.legend(loc='upper right', handles=handles, frameon=False, title=text)
    legend._legend_box.align = 'right'

    plt.setp(legend.get_texts(), color=cmap(0.98))
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 25
    plt.rcParams['figure.autolayout'] = True

    if(bnds):
        cntr  =  np.array([np.mean(curr_event[:,0]), np.mean(curr_event[:, 1])])
        cntr_std =  np.array([np.std(curr_event[:,0]), np.std(curr_event[:, 1])])
        std = max(cntr_std)
        ax.set_xlim(-40,60)
        ax.set_ylim(10,30)
    ax.set_facecolor(cmap(c_cut))

    if(NOAXIS):
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        g.ax_marg_y.get_yaxis().set_visible(False)
        g.ax_marg_x.get_xaxis().set_visible(False)
    
    if(EMD):
        ax.text(.98, .02,
                s=r"EMD: $\mathbf{%.3f}$"%distance, 
                transform=ax.transAxes,
                horizontalalignment='right',
                verticalalignment='bottom',)



#     ax.text(-.005 , 0.5, s=r'\textbf{t-SNE Embedding} of Latent Representations',
#             horizontalalignment='right',
#             verticalalignment='center',
#             rotation='vertical',
#             transform=ax.transAxes)
    
    return g


