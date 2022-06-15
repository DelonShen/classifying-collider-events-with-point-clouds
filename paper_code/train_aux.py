import os

#TEMP
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

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

train_bst = False

#TESTING
# EPOCHS = 2
# filename = '../data/data50k_raw_combined_atlas_cut_small.pkl'
# num_round = 1
######


# In[3]:
def human_format(num):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    # add more suffixes if you need them
    return '%.0f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])


PI = experiment.Experimenter(filename)
PI.fromSaved()

to_train = ['nested_concat_general']
for nm in to_train:
    print('RIGHT NOW: %s'%nm)
    PI.data_loader(nm, gen_multijet_to_inv_dataset, class_weight_invariant, tf.constant, aux_params=dict(dR_keep=False, multijet_n=1))
    PI.train_classifier(nm, model_params_dict[nm], epochs=EPOCHS)
    print('###')
    PI.save_experimenter()