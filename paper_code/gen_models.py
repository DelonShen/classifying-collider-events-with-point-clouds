
# coding: utf-8

# In[1]:


# # Train the Models
# This first block of code trains the models

# In[2]:


import os

#to get latex to work 
os.environ['PATH'] = "%s:/usr/local/cuda-11.2/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/opt/anaconda/bin:/home/delon/texlive/bin/x86_64-linux:/home/delon/.local/bin:/home/delon/bin"%os.environ['PATH']

import sys
sys.path.insert(1, '../')

import experiment 
from utils import *
from Architectures import *
import seaborn as sns

import random




suppress_warnings()
EPOCHS = 512

os.environ['PYTHONHASHSEED']=str(0)
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)


filename = '../data/data100k_raw_combined_atlas_cut.pkl'
num_round = None

#TESTING
#EPOCHS = 2
#filename = '../data/data50k_raw_combined_atlas_cut_small.pkl'
#num_round = 1
######


# In[3]:


SUFFIX = ''
PI = experiment.Experimenter(filename)
PI.fromSaved()

##IF FROM SAVED
#############

to_train = ['particlewise',
            'nested_concat',
            'nested_concat_general',
            'tripletwise',
            'pairwise',
            'pairwise_nl',
            'pairwise_nl_iter',
            'naivednn',]

for nm in to_train:
    print('RIGHT NOW: %s'%nm)
    PI.data_loader(nm, gen_multijet_to_inv_dataset, class_weight_invariant, tf.constant, aux_params=dict(dR_keep=False, multijet_n=1))
    PI.train_classifier(nm, model_params_dict[nm], epochs=EPOCHS)
    print('###')
    PI.save_experimenter(suffix=SUFFIX)

print('DNN Classifier')
PI.data_loader('dnn', gen_dataset_high_level, class_weight_invariant, tf.constant)
PI.train_classifier('dnn', model_params_dict[nm] , use_weights_during_fit = True, epochs=EPOCHS)
print('###')
PI.save_experimenter(suffix=SUFFIX)
