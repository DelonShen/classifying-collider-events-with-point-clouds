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


filename = '../data/data100k_raw_combined_atlas_cut.pkl'
n_experiments = 8
SUFFIX = '_latent_dim_edge_AeqB_only_1_2'


#TESTING ######
#EPOCHS = 2
#filename = '../data/data50k_raw_combined_atlas_cut_small.pkl'
#n_experiments = 2
#SUFFIX = '_latent_dim_edge_TEST'
###########

PI = experiment.Experimenter(filename)

AUCs=[]
nn_AUC = 0
print('DNN Classifier')
PI.data_loader('dnn', gen_dataset_high_level, class_weight_invariant, tf.constant)
PI.train_classifier('dnn', {'width':256,'depth':3,'num_classes':2} , use_weights_during_fit = True, epochs=EPOCHS)
print('###')


PI.data_loader('pairwise', gen_multijet_to_inv_dataset, class_weight_invariant, tf.constant, aux_params=dict(dR_keep=False, multijet_n=1))
for i in scan:
    curr_auc = []
    print("\t\tLATENT DIM", 2**i)
    for j in range(n_experiments):
        tf.random.set_seed(42+j)
        np.random.seed(42+j)
        random.seed(42+j)

        PI.train_classifier('pairwise', {'depth':5, 'ec_widths':(64,128,256,128,2**i), 'width':64}, epochs=EPOCHS, seed=42+j, patience=8)
        #PI.save_experimenter(suffix=SUFFIX)

        tail_string = PI.get_tail_string({'depth':5, 'ec_widths':(64,128,256,128,2**i), 'width':64})
        classifier_name = '%s_%s'%('pairwise', tail_string)
        auc = PI.get_ROC('pairwise', {'depth':5, 'ec_widths':(64,128,256,128,2**i), 'width':64})[-1]
        curr_auc.append(auc)
        if(j==0):
            n_params.append(human_format(PI.models[classifier_name].count_params()))
        del PI.models[classifier_name]
        del PI.perf[classifier_name]
    AUCs.append(curr_auc)
    print('\t\t\t', 2**i, curr_auc)

nn_AUC = PI.get_ROC('dnn', {'width':256,'depth':3,'num_classes':2})[-1]


meanst = np.array([np.mean(AUC) for AUC in AUCs])
sdt = np.array([np.std(AUC) for AUC in AUCs])


table_file = open('figures/latent_scan_table.tex', 'w+')
for a in scan:
    table_file.write('$2^%d$ & $%.3f \pm %.3f$ & %s\\\\\n'%(a, meanst[a], sdt[a], n_params[a]))
table_file.close()

data_filename = 'figures/latent_scan_auc' 
data_file = open(data_filename, 'wb')
pickle.dump((AUCs, nn_AUC), data_file)
data_file.close()

