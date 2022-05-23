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
#EPOCHS = 1
#filename = '../data/data50k_raw_combined_atlas_cut_small.pkl'
#num_round = 1
######




SUFFIX = ''
PI = experiment.Experimenter(filename)

to_train = ['particlewise',
            'pairwise',
            'tripletwise',
            'pairwise_nl',
            'pairwise_nl_iter',
            'nested_concat',
            'naivednn']

for nm in to_train:
    print('RIGHT NOW: %s'%nm)
    PI.data_loader(nm, gen_multijet_to_inv_dataset, class_weight_invariant, tf.constant, aux_params=dict(dR_keep=False, multijet_n=1))
    PI.train_classifier(nm, model_params_dict[nm], epochs=EPOCHS)
    print('###')

print('DNN Classifier')
PI.data_loader('dnn', gen_dataset_high_level, class_weight_invariant, tf.constant)
PI.train_classifier('dnn', model_params_dict[nm] , use_weights_during_fit = True, epochs=EPOCHS)
print('###')




##### Generating graphs
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

filename = 'data/data100k_raw_combined_atlas_cut.pkl'

def human_format(num):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    # add more suffixes if you need them
    return '%.0f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])




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


bst_filename = 'models/'+filename.split('.')[0].split('/')[-1]+'BDT.json'
bst = None
import pickle
bst = xgb.train(param, xg_train, num_round)


yhat_test = bst.predict(xg_test).reshape(yo_test.shape[0], 2)
yhat_test = np.array([true for (true, false) in yhat_test])
yop_test  = np.array([true for (true, false) in y_test])

from sklearn import metrics

fpr, tpr, thresholds = metrics.roc_curve(yop_test, yhat_test)
auc = metrics.auc(fpr, tpr)

print(auc)


import bisect 

location = bisect.bisect_left(list(reversed(thresholds)), 0.5)
print('At 0.5 threshold we have BDT signal efficiency %.3f'%list(reversed(tpr))[location-1])




models_to_plot = [PI.model_name_to_model(model_name) for model_name in list(PI.models.keys())]
model_params_to_plot = [model_params_dict[PI.model_name_to_model(model_name)] for model_name in list(PI.models.keys())]
model_params_to_plot


fig, ax = PI.plot_multiple_sorted_by_AUC(models_to_plot, model_params_to_plot)
colormap = sns.cubehelix_palette(start=26/10, light=.97, as_cmap=True)
ax.plot(tpr, 1/fpr, label=r'AUC: %.3f for %s'%(auc, 'BDT + ATLAS Features'), color=colormap(0.33))                                         


ax.get_legend().remove()
ax.legend(loc='upper right', frameon=False, labelspacing=2.0)
handles, labels = ax.get_legend_handles_labels()
labels, handles = list(zip(*reversed(sorted(zip(labels, handles)))))
ax.legend(handles, labels, loc='upper right', frameon=False)

for txt in ax.texts:
    txt.set_visible(False)

annotation_string = r'\textbf{ROC Curve for Event Classification}'
annotation_string += '\n'
annotation_string += r'$t\overline{t}(H\rightarrow\tau\tau)$ and $t\overline{t}(t\rightarrow \tau\nu b)$'
annotation_string += '\n'
annotation_string += r'\textsc{MadGraph 5}+\textsc{Pythia} 8+\textsc{Delphes}'
annotation_string += '\n'
annotation_string += r'Anti-Kt with $R=0.4$, $\sqrt{s} = 14$'
ax.text(.05,1, annotation_string)
plt.gcf().set_size_inches(8.5, 8.5)
fig.savefig('figures/roc_curves.pdf')




#generate table of data for performance at signal efficiency
import bisect 
location = bisect.bisect_left(list(reversed(thresholds)), 0.5)

table_file = open('figures/performance_table.tex', 'w+')
table_data = [(PI.get_ROC(classifier_key, param)) for (classifier_key, param) in zip(models_to_plot, model_params_to_plot)]

c_names = classifiers_name
for (fpr, tpr, thresholds, auc), key, params in zip(table_data, models_to_plot, model_params_to_plot):
    location_0 = bisect.bisect_left(tpr, 0.7)
    location = bisect.bisect_left(tpr, 0.3)
    c_model = PI.models['%s_%s'%(key, PI.get_tail_string(params))]
    NPARAMS = human_format(c_model.count_params())
    table_file.write('%s & %.3f & %s & %.3f & %.3f\\\\\n'%(c_names[key], auc, NPARAMS,
                                              1/fpr[location_0-1], (tpr)[location_0-1]/fpr[location_0-1]))
    
yhat_test = bst.predict(xg_test).reshape(yo_test.shape[0], 2)
yhat_test = np.array([true for (true, false) in yhat_test])
yop_test  = np.array([true for (true, false) in y_test])
from sklearn import metrics
fpr, tpr, thresholds = metrics.roc_curve(yop_test, yhat_test)
auc = metrics.auc(fpr, tpr)
location_0 = bisect.bisect_left(tpr, 0.7)
location = bisect.bisect_left(tpr, 0.3)
table_file.write('%s & %.3f & -  & %.3f & %.3f\\\\\n'%('BDT + ATLAS Features', auc, 1/fpr[location_0-1], (tpr)[location_0-1]/fpr[location_0-1]))

table_file.close()




####GENERATING TSNE PLOT
from tqdm import trange

from scipy.stats import gaussian_kde 
from scipy.stats import kendalltau
import seaborn as sns

tail_string = PI.get_tail_string(model_params_dict['pairwise'])
pairwise_model = PI.models['%s_%s'%('pairwise', tail_string)]
latent_getter = LatentGetter(pairwise_model.layers[0:3], condensed=True)

X_test, y_test = PI.get_test_dataset('pairwise')

n_cut = int(len(X_test)*0.5)
indices = np.random.permutation(len(X_test))
cut = np.s_[indices[:n_cut]]
latent_reps = latent_getter.predict(X_test.numpy()[cut])

latent_label = y_test.numpy()[cut]

from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances

distance_matrix = pairwise_distances(latent_reps, latent_reps, metric='cosine', n_jobs=-1)
latent_reps_embedded_tsne = TSNE(metric="precomputed", n_components=2, learning_rate='auto', 
                                  verbose=2, perplexity=50, 
                                 n_iter=2000, n_jobs=-1)
latent_reps_embedded = latent_reps_embedded_tsne.fit_transform(distance_matrix)




c_cut = 5
cmap = sns.cubehelix_palette(start=26/10, light=.97, as_cmap=True)

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 16
plt.rcParams['figure.autolayout'] = False
plt.rcParams['text.usetex'] = True

curr_event = latent_reps_embedded
ttH_loc = np.array([curr_event[i] for i in range(len(latent_label)) if latent_label[i][1]==1])
g = sns.jointplot(x=ttH_loc[:,0], y = ttH_loc[:,1], color=cmap(100), space=0, label='ttH jets',
                  cmap=cmap, kind='kde', height=10, fill=True, cut=c_cut,
                 marginal_kws={'linewidth': 0.0, 'alpha':1.0})

COL2 = '#000000'
linew = 0.6
nttH_loc = np.array([curr_event[i] for i in range(len(latent_label)) if latent_label[i][0]==1])
sns.kdeplot(x=nttH_loc[:,0], y = nttH_loc[:,1], shade=False, label=r'ttbar jets',color=COL2, linewidths=linew, cut=c_cut, 
            levels=10, ax=g.ax_joint)

sns.kdeplot(nttH_loc[:,0], ax=g.ax_marg_x, color=COL2, lw=linew)
sns.kdeplot(y=nttH_loc[:,1], ax=g.ax_marg_y, color=COL2, lw=linew)

ax = g.ax_joint

import matplotlib.patches as  mpatches
import matplotlib.lines as  mlines

handles = [mpatches.Patch(facecolor=cmap(100), label=r'$tt(H\rightarrow\tau\tau)$ Events'),
           mlines.Line2D([], [], color=COL2, label=r'$t\overline{t}$ Events', lw=linew)]
legend = ax.legend(loc='upper left', handles=handles, frameon=False, title=r'\textbf{Latent Representation} in Pairwise Architecture')
legend._legend_box.align = 'left'
plt.setp(legend.get_texts(), color=cmap(0.98))
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 16
plt.rcParams['figure.autolayout'] = True



ax.set_facecolor(cmap(c_cut))
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
g.ax_marg_y.get_yaxis().set_visible(False)
g.ax_marg_x.get_xaxis().set_visible(False)


plt.savefig('figures/tsne_densities.pdf')

