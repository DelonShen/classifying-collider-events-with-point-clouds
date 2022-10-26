# import numpy for Scientific computations
import numpy as np
from utils import *

# import machine learning libraries
import xgboost as xgb

# import packages for hyperparameters tuning
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
MAX_EVAL = 256
filename = '../data/data80k_raw_combined_atlas_cut.pkl'

#filename = 'data/data50k_raw_combined_atlas_cut_small.pkl'
#MAX_EVAL = 100



import experiment
PI = experiment.Experimenter(filename)

X_train, y_train = gen_dataset_high_level(PI.events_train, PI.events_oup_train, PI.events_tag_train) 
X_test, y_test = gen_dataset_high_level(PI.events_test, PI.events_oup_test, PI.events_tag_test) 

X_train = np.array(X_train)
yo_train = np.array([np.argmax(y) for y in y_train])
yo_test = np.array([np.argmax(y) for y in y_test])
X_test = np.array(X_test)

xg_train = xgb.DMatrix(X_train, label=yo_train)
xg_test = xgb.DMatrix(X_test, label=yo_test)

space={'max_depth': hp.quniform("max_depth", 3, 18, 1),
        'gamma': hp.uniform ('gamma', 1,9),
        'reg_alpha' : hp.quniform('reg_alpha', 40,180,1),
        'reg_lambda' : hp.uniform('reg_lambda', 0,1),
        'colsample_bytree' : hp.uniform('colsample_bytree', 0.5,1),
        'min_child_weight' : hp.quniform('min_child_weight', 0, 10, 1),
        'n_estimators': hp.quniform('n_estimators', 100, 1000, 1),
        'eta': hp.quniform('eta', 0.025, 0.5, 0.025),
        'seed': 0
    }


def objective(space):
    param =         {'max_depth': int(space['max_depth']), 'gamma' : space['gamma'],
                    'reg_alpha' : int(space['reg_alpha']),'min_child_weight':int(space['min_child_weight']),
                    'colsample_bytree':int(space['colsample_bytree']), 'eta':space['eta']}

    param['objective'] = 'multi:softprob'
    param['num_class'] = 2

    bst = xgb.train(param, xg_train, int(space['n_estimators']))

    yhat_test = bst.predict(xg_test).reshape(yo_test.shape[0], 2)
    yhat_test = np.array([true for (true, false) in yhat_test])
    yop_test  = np.array([true for (true, false) in y_test])
    
    from sklearn import metrics
    
    fpr, tpr, thresholds = metrics.roc_curve(yop_test, yhat_test)
    auc = metrics.auc(fpr, tpr)
    loss = 1 - auc
    print('AUC is', auc)
    return {'loss': loss, 'status': STATUS_OK}
    


trials = Trials()
best_hyperparams = fmin(fn = objective,
                        space = space,
                        algo = tpe.suggest,
                        max_evals = MAX_EVAL,
                        trials = trials,
                        verbose=2)


print("The best hyperparameters are : ","\n")
print(best_hyperparams)
