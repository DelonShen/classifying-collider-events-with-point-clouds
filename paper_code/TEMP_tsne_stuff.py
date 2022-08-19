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

do_tsne = False
#TESTING
#EPOCHS = 2
#filename = '../data/data50k_raw_combined_atlas_cut_small.pkl'
#num_round = 1
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




# # Now we're gonna generate some graphs and get the performance of the archictures

# In[4]:


import seaborn as sns

matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['axes.unicode_minus'] = False

matplotlib.rcParams['text.latex.preamble'] = [
    r'\usepackage{amsmath}',
    r'\usepackage{amssymb}',
    r'\usepackage{hyperref}',
    ] 

suppress_warnings()
EPOCHS = 64

def human_format(num):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    # add more suffixes if you need them
    return '%.0f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])


# In[5]:


import xgboost as xgb

X_train, y_train = gen_dataset_high_level(PI.events_train, PI.events_oup_train, PI.events_tag_train) 
X_test, y_test = gen_dataset_high_level(PI.events_test, PI.events_oup_test, PI.events_tag_test) 

X_train = np.array(X_train)
yo_train = np.array([np.argmax(y) for y in y_train])
yo_test = np.array([np.argmax(y) for y in y_test])
X_test = np.array(X_test)

xg_train = xgb.DMatrix(X_train, label=yo_train)
xg_test = xgb.DMatrix(X_test, label=yo_test)


# setup parameters for xgboost
param = {'colsample_bytree': 0.7729268575934765, 'eta': 0.25, 'gamma': 1.002343020792451, 'max_depth': 10, 'min_child_weight': 9, 'n_estimators': 530, 'reg_alpha': 41.0, 'reg_lambda': 0.8554269844258477} #THESE SELECTED as optimal BY HYPEROPT 

# use softmax multi-class classification
param['objective'] = 'multi:softprob'
param['num_class'] = 2

if(num_round == None):
    num_round = param['n_estimators']



bst_filename = 'models/'+filename.split('/')[-1].split('.')[0]+'BDT.json'
bst = None
import pickle
if(train_bst):
    bst = xgb.train(param, xg_train, num_round)
    pickle.dump(bst, open(bst_filename, 'wb'))
else:
    bst = pickle.load(open(bst_filename, 'rb'))


yhat_test = bst.predict(xg_test).reshape(yo_test.shape[0], 2)
yhat_test = np.array([true for (true, false) in yhat_test])
yop_test  = np.array([true for (true, false) in y_test])


# In[6]:


from sklearn import metrics

fpr, tpr, thresholds = metrics.roc_curve(yop_test, yhat_test)
auc = metrics.auc(fpr, tpr)

import bisect 

location = bisect.bisect_left(list(reversed(thresholds)), 0.5)
print('At 0.5 threshold we have BDT signal efficiency %.3f'%list(reversed(tpr))[location-1])
print(auc)


