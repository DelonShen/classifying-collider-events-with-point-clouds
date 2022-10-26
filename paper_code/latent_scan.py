
# coding: utf-8

# In[1]:


import sys
sys.path.insert(1, '../')

import experiment
import pickle
from utils import *
from Architectures import *
import random, os
suppress_warnings()
EPOCHS = 256


def countp(model, params):
    tmp = model(**params)
    tmp.build(input_shape=(1,15,7))
    return tmp.count_params()


scan = range(7)
n_params = []

os.environ['PYTHONHASHSEED']=str(0)
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

def human_format(num):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    # add more suffixes if you need them
    return '%.0f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])


filename = '../data/data80k_raw_combined_atlas_cut.pkl'
n_experiments = 8
SUFFIX = '_latent_dim_edge_AeqB_only_1_2'


#TESTING ######
#EPOCHS = 2
#filename = '../data/data50k_raw_combined_atlas_cut_small.pkl'
#n_experiments = 2
#SUFFIX = '_latent_dim_edge_TEST'
###########


# In[ ]:


PI = experiment.Experimenter(filename)
from datetime import datetime

AUCs=[]
e03i = []
e03 = []
e07i = []
e07 = []
nn_AUC = 0
print('DNN Classifier')
PI.data_loader('dnn', gen_dataset_high_level, class_weight_invariant, tf.constant)
PI.train_classifier('dnn', {'width':256,'depth':3,'num_classes':2} , use_weights_during_fit = True, epochs=EPOCHS)
print('###')

import bisect 

temp_log_file = open('figures/latent_scan_table_temp_log', 'w+')
temp_log_file.close()

PI.data_loader('pairwise', gen_multijet_to_inv_dataset, class_weight_invariant, tf.constant, aux_params=dict(dR_keep=False, multijet_n=1))
for i in scan:
    curr_auc = []
    print("\t\tLATENT DIM", 2**i)
    for j in range(n_experiments):
        tf.random.set_seed(42+j)
        np.random.seed(42+j)
        random.seed(42+j)
        tail_string = PI.get_tail_string({'depth':5, 'ec_widths':(64,128,256,128,2**i), 'width':64})
        classifier_name = '%s_%s'%('pairwise', tail_string)

        if(classifier_name in PI.models):
            del PI.models[classifier_name]
            del PI.perf[classifier_name]

        PI.train_classifier('pairwise', {'depth':5, 'ec_widths':(64,128,256,128,2**i), 'width':64}, epochs=EPOCHS, seed=42+j)


        fpr, tpr, thresholds, auc = PI.get_ROC('pairwise', {'depth':5, 'ec_widths':(64,128,256,128,2**i), 'width':64})
        curr_auc.append(auc)
        location_0 = bisect.bisect_left(tpr, 0.7)
        location = bisect.bisect_left(tpr, 0.3)
        e03i.append(1/fpr[location-1])
        e03.append((tpr)[location-1]/fpr[location-1])
        e07i.append(1/fpr[location_0-1])
        e07.append((tpr)[location_0-1]/fpr[location_0-1])
        if(j==0):
            n_params.append(human_format(PI.models[classifier_name].count_params()))
    AUCs.append(curr_auc)
    print('\t\t\t', 2**i, curr_auc)
    temp_log_file = open('figures/latent_scan_table_temp_log', 'a+')
    temp_log_file.write('%s\t %d\n'%(datetime.now(), 2**i))
    temp_log_file.close()

nn_AUC = PI.get_ROC('dnn', {'width':256,'depth':3,'num_classes':2})[-1]


# In[ ]:


def mean_std(lst,a):
    return np.array([np.mean(curr) for curr in lst])[a], np.array([np.std(curr) for curr in lst])[a]

table_file = open('figures/latent_scan_table.tex', 'w+')
for a in scan:
    table_file.write('$2^%d$ & $%.3f \pm %.3f$ & $%.1f\pm %.2f$ & $%.1f\pm %.2f$ & $%.1f\pm %.2f$ & $%.1f\pm %.2f$ & %s\\\\\n'%(a, *mean_std(AUCs,a-1),
                                                                                                                                *mean_std(e03i,a),
                                                                                                                                *mean_std(e03,a),
                                                                                                                                *mean_std(e07i,a),
                                                                                                                                *mean_std(e07,a), 
                                                                                                                                n_params[a]))
table_file.close()

data_filename = 'figures/latent_scan_auc' 
data_file = open(data_filename, 'wb')
pickle.dump((AUCs, nn_AUC, e03, e03i, e07i, e07, n_params ), data_file)
data_file.close()

