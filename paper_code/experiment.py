from utils import *
from tensorflow.keras.callbacks import EarlyStopping
import bisect 

from sklearn import metrics
import pickle
import itertools
from Architectures import *
import seaborn as sns

BATCH_SIZE = 32

matplotlib.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 20
plt.rcParams['figure.autolayout'] = True



class Experimenter:
    
    def __init__(self, filename, split=0.7):
        self.classifiers = classifiers
        self.classifiers_name = classifiers_name
        self.perf = {}
        print('Initializing Experimenter')
        self.filename = filename
        print('\tLoading Data from %s'%(filename))
        self.events, self.events_oup, self.events_tag = load_data_from_file(filename)
        print('\tData Loaded')
        print('\tCreating Splits')
        self.training_idx, self.test_idx = train_test_split(len(self.events), split=split)
        self.events_train, self.events_test = self.get_split(self.events)
        self.events_oup_train, self.events_oup_test = self.get_split(self.events_oup)
        self.events_tag_train, self.events_tag_test = self.get_split(self.events_tag)
        print('\tSplits Created')
        self.models = dict()
        self.datasets = dict()
        self.model_directories = dict()
        print('Done initalizing')

    def model_name_to_model(self, model_name):
        """For example inv_from_triplet_100_3_256 -> inv_from_triplet"""
        model_name = model_name.split('_')
        model_name = [word for word in model_name if not any(chr.isdigit() for chr in word)]
        return '_'.join(model_name)

    def model_name_to_params(self, model_name):
        model_name = model_name.split('_')
        model_name = [word for word in model_name if any(chr.isdigit() for chr in word)]
        return tuple(eval(elem) for elem in model_name)

    def load_model_directories(self, filename):
        experimenter_file = open(filename, 'rb')
        _, _, _, _, _, _, _, self.model_directories = pickle.load(experimenter_file)

    def update_data(self, suffix=''):
        self.events, self.events_oup, self.events_tag = load_data_from_file(self.filename)
        print('\tData Loaded')
        print('\tCreating Splits')
        self.events_train, self.events_test = self.get_split(self.events)
        self.events_oup_train, self.events_oup_test = self.get_split(self.events_oup)
        self.events_tag_train, self.events_tag_test = self.get_split(self.events_tag)
        print('now saving paramters of experimenter')
        experimenter_filename  ='/data/delon/experimenter/'+self.filename.split('/')[-1].split('.')[0]+suffix
        experimenter_file = open(experimenter_filename, 'wb')
        pickle.dump((self.filename, self.events, self.events_oup, self.events_tag, self.training_idx, self.test_idx, self.datasets, self.model_directories), experimenter_file)
        experimenter_file.close()
        print('saved experimenter at', experimenter_filename)


    def fromSaved(self, suffix=''):

        experimenter_filename  ='/data/delon/experimenter/'+self.filename.split('/')[-1].split('.')[0]+suffix
        print('Loading Experimenter from Saved Experimenter at',experimenter_filename)
        experimenter_file = open(experimenter_filename, 'rb')
        self.filename, self.events, self.events_oup, self.events_tag, self.training_idx, self.test_idx, self.datasets, self.model_directories = pickle.load(experimenter_file)
        experimenter_file.close()
        print('Experimenter Loaded')
        print('Getting split')
        self.events_train, self.events_test = self.get_split(self.events)
        self.events_oup_train, self.events_oup_test = self.get_split(self.events_oup)
        self.events_tag_train, self.events_tag_test = self.get_split(self.events_tag)
        self.models = dict()
        print('Split Stored')
        base_filename = self.filename.split('/')[-1].split('.')[0]
        print('Loading models')
        print(self.model_directories)
        for model_name in self.model_directories.keys():
            model_filename = self.model_directories[model_name]
            weights = np.array([*self.datasets[self.model_name_to_model(model_name)+'_weight']])
            loss = weightedLoss(tf.keras.losses.categorical_crossentropy, weights)
            tmp = {}
            tmp['['] = '('
            tmp[']'] = ')'
            model_filename = ''.join([a if (a not in ['[',']']) else tmp[a] for a in model_filename])
            self.models[model_name] = self.load_model_weights(model_name, self.model_name_to_model(model_name), suffix=suffix)#, custom_objects={"_my_ragged_tensor_lossFunc":loss})
            print('Loaded %s from %s'%(model_name, model_filename))

    


    def get_split(self, dataset):
        return dataset[self.training_idx], dataset[self.test_idx]

    def data_loader_sequential(self, data_key, prev_classifier, params, y_data_generator, weight_generator, y_tensorizer):
        X_prev_train, _ = self.datasets[prev_classifier+'_train']
        X_prev_test, _  = self.datasets[prev_classifier+'_test']

        tail_string = self.get_tail_string(params)
        classifier_name = '%s_%s'%(prev_classifier, tail_string)

        if(classifier_name not in self.models):
            print('Warning: previous classifier not made yet. Temporarily creating')
            self.models[classifier_name] = self.classifiers[classifier_key](**params)

        prev_classifier = self.models[classifier_name]

        X_train = prev_classifier(X_prev_train)
        X_test  = prev_classifier(X_prev_test)

        y_train = y_data_generator(self.events_oup_train, self.events_tag_train)
        y_test = y_data_generator(self.events_oup_test, self.events_tag_test)

        y_train = y_tensorizer(y_train)
        y_test  = y_tensorizer(y_test)

        self.datasets[data_key+'_train'] = (X_train, y_train)
        self.datasets[data_key+'_test'] = (X_test, y_test)

        self.datasets[data_key+'_weight']  = weight_generator(y_train)


    def data_loader_multi_input(self, data_key, data_generators, weight_generator, y_tensorizer, aux_params=[]):
        X_train = [data_generators[i](self.events_train, self.events_oup_train, self.events_tag_train, **aux_params[i]) for i in range(len(data_generators))]
        X_test  = [data_generators[i]( self.events_test,  self.events_oup_test,  self.events_tag_test, **aux_params[i])    for i in range(len(data_generators))]



        y_train = X_train[0][1]
        X_train = [X[0] for X in X_train]

        y_test  = X_test[0][1]
        X_test  = [X[0] for X in X_test]

        self.datasets[data_key+'_weight']  = weight_generator(y_train)

        X_train = [tf.constant(X) for X in X_train]
        y_train = y_tensorizer(y_train)
        X_test  = [tf.constant(X) for X in X_test]
        y_test  = y_tensorizer(y_test)

        y_train = tf.cast(y_train, tf.float32)
        y_test = tf.cast(y_test, tf.float32)

        self.datasets[data_key+'_train'] = (X_train, y_train)
        self.datasets[data_key+'_test'] = (X_test, y_test)


    def data_loader(self, data_key, data_generator, weight_generator, y_tensorizer, aux_params = dict()):
        X_train, y_train = data_generator(self.events_train, self.events_oup_train, self.events_tag_train, **aux_params)
        X_test, y_test = data_generator(self.events_test, self.events_oup_test, self.events_tag_test, **aux_params)

        self.datasets[data_key+'_weight']  = weight_generator(y_train)

        X_train = tf.constant(X_train)
        y_train = y_tensorizer(y_train)
        X_test  = tf.constant(X_test)
        y_test  = y_tensorizer(y_test)

#        y_train = tf.cast(y_train, tf.float32)
#        y_test = tf.cast(y_test, tf.float32)

        self.datasets[data_key+'_train'] = (X_train, y_train)
        self.datasets[data_key+'_test'] = (X_test, y_test)


    def get_tail_string(self, params):
        tail_string = ''
        for param in params:
            if(type(params[param]) == type(True)):
                tail_string += str(int(params[param])) + '_'
                continue
            tail_string += str(params[param]) + '_'
        return tail_string[:-1]


    def train_classifier(self, 
            classifier_key, 
            params, 
            epochs=10, 
            learning_rate = 1e-3, 
            loss = tf.keras.losses.CategoricalCrossentropy(),
            use_weights_during_fit = False, 
            use_optim=True,
            seed=42,
            patience=32):

        tf.random.set_seed(seed)
        X_train, y_train = self.datasets[classifier_key + '_train']
        dataset_train = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        dataset_train = dataset_train.batch(BATCH_SIZE)
        print('tf.data.datset created for training data') 

        weight_for_0, weight_for_1 = self.datasets[classifier_key + '_weight']
        weights = {0:weight_for_0, 1:weight_for_1}

        tail_string = self.get_tail_string(params)
        classifier_name = '%s_%s'%(classifier_key, tail_string)

        if(classifier_name not in self.models):
            print('Model not yet created, creating new model')
            self.models[classifier_name] = self.classifiers[classifier_key](**params)

        classifier = self.models[classifier_name]

        opt = tf.keras.optimizers.Adam(amsgrad = True, learning_rate = learning_rate)
        if(use_optim):
            classifier.compile(loss = loss, optimizer=opt)

        callbacks = [EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)]
        if(use_weights_during_fit):
            classifier.fit(X_train, y_train, verbose=2,
                    epochs=epochs, class_weight = weights,
                    validation_split=0.3, callbacks=callbacks)
        else:
            classifier.fit(X_train, y_train, verbose=2,
                    epochs=epochs,
                    validation_split=0.3, callbacks=callbacks)

    def systematics_test_roc_auc(self,
            classifier_key,
            params,
            data_generator,
            aux_params={}):
        X_train, y_train = data_generator(self.events_train, self.events_oup_train, self.events_tag_train, systematics_test=True, **aux_params)
        X_test, y_test = data_generator(self.events_test, self.events_oup_test, self.events_tag_test, systematics_test=True, **aux_params)

        X_train = tf.constant(X_train)
        y_train = tf.constant(y_train)
        X_test  = tf.constant(X_test)
        y_test  = tf.constant(y_test)

        tail_string = self.get_tail_string(params)
        classifier_name = '%s_%s'%(classifier_key, tail_string)
        classifier = self.models[classifier_name]

        print('currently on', classifier_name)

        yhat_test = classifier.predict(X_test)
        y_test = y_test.numpy()

        yhat_test = np.array([true for (true,false) in yhat_test])
        y_test    = np.array([true for (true,false) in y_test   ])


        fpr, tpr, thresholds = metrics.roc_curve(y_test, yhat_test)
        auc = metrics.auc(fpr, tpr)

        return fpr, tpr, thresholds, auc
    



    def get_ROC(self,
            classifier_key,
            params,
            ideal=None):
        '''get ROC and AUC for EVENT CLASSIFICATION
        ideal means 6jets 2b 2tau'''
        
        print('getting ROC for',classifier_key)
                     
        tail_string = self.get_tail_string(params)
        classifier_name = '%s_%s'%(classifier_key, tail_string)
        classifier = self.models[classifier_name]
        
        if(classifier_name in self.perf):
            print('pog')
            return self.perf[classifier_name]

        X_test, y_test = self.datasets[classifier_key + '_test']



        print('currently on', classifier_name)
#        classifier = self.load_model_special(classifier_key, params)

        yhat_test = classifier.predict(X_test)
        y_test = y_test.numpy()

        yhat_test = np.array([true for (true,false) in yhat_test])
        y_test    = np.array([true for (true,false) in y_test   ])


        fpr, tpr, thresholds = metrics.roc_curve(y_test, yhat_test)
        auc = metrics.auc(fpr, tpr)

        self.perf[classifier_name] = (fpr, tpr, thresholds, auc)
        return fpr, tpr, thresholds, auc
    
    def plot_ROC(self, 
            classifier_key,
            params,
            ax=None,
            ideal=None):
        '''plots ROC and AUC for EVENT CLASSIFICATION'''

        fpr, tpr, thresholds, auc = self.get_ROC(classifier_key, params, ideal=ideal)
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.size'] = 20
        location_0 = bisect.bisect_left(tpr, 0.7)
        e7 = 1/fpr[location_0-1]
        if(ax != None):
            ax.plot(tpr, 1/fpr, **lstyle[classifier_key], label=r'%.5f %s'%(e7, classifiers_name[classifier_key]))
            ax.set_yscale('log')
            return

        plt.figure(figsize=(8, 10), dpi=80)
        plt.plot(tpr, 1/fpr, color='black', label=r'AUC: %.5f'%(auc))
        plt.xlim([0.0, 1.0])
        ax.set_xlabel(r'$\epsilon_s$')
        ax.set_ylabel(r'$1/\epsilon_b$')
        plt.legend(loc='lower right', frameon=False)
        tail_string = self.get_tail_string(params)
        classifier_name = '%s_%s'%(classifier_key, tail_string)
        plt.savefig('figures/'+classifier_name + ".pdf")

    def plot_multiple(self,
            classifier_keys,
            params,
            ideal=None):

        models = zip(classifier_keys, params)

        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.size'] = 20
    
        fig, ax = plt.subplots(figsize=(7,10), dpi=80)
        for (classifier_key, param) in models:
            self.plot_ROC(classifier_key, param, ax=ax, ideal=ideal)

        colormap = sns.cubehelix_palette(start=26/10, light=.97, as_cmap=True)
        colors = [colormap(i) for i in np.linspace(.3, .98,len(ax.lines))]
#         for i,j in enumerate(ax.lines):
#             j.set_color(colors[i])


        ax.set_xlim([0,1])
        ax.set_xlabel(r'$\epsilon_s$')
        ax.set_ylabel(r'$1/\epsilon_b$')
        ax.legend(loc='lower right', frameon=False)
#        handles, labels = ax.get_legend_handles_labels()
#        ax.legend(handles[::-1], labels[::-1], loc='lower right', frameon=False)

        return fig, ax

    def plot_multiple_sorted_by_AUC(self,
            classifier_keys,
            params,
            ideal=None):

        print('alright we\'re gonna start look at', classifier_keys)
        models = zip(classifier_keys, params)
        AUCs = [self.get_ROC(*model, ideal=ideal)[-1] for model in models]
        AUCs, classifier_keys, params = list(zip(*sorted(zip(AUCs, classifier_keys, params))))
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.size'] = 20

        return self.plot_multiple(classifier_keys, params, ideal=ideal)

    def save_experimenter(self, suffix=''):
        print('first saving models')
        self.save_models(suffix=suffix)
        print('now saving paramters of experimenter')
        experimenter_filename  ='/data/delon/experimenter/'+self.filename.split('/')[-1].split('.')[0]+suffix
        experimenter_file = open(experimenter_filename, 'wb')
        pickle.dump((self.filename, self.events, self.events_oup, self.events_tag, self.training_idx, self.test_idx, self.datasets, self.model_directories), experimenter_file)
        experimenter_file.close()
        print('saved experimenter at', experimenter_filename)

    def save_models(self, suffix=''):
        base_filename = self.filename.split('/')[-1].split('.')[0]
        for model_name in self.models.keys():
            print('currently on', model_name)
            model = self.models[model_name]
            model_filename = 'models/'+base_filename+'_'+model_name+suffix
            if(model_name in self.model_directories.keys()): #e.g. exsisted when loaded from file
                print('\tthis one already saved, skipped')
                continue
            model.save(model_filename)
            self.model_directories[model_name] = model_filename
            print('%s is saved in %s'%(model_name, model_filename))

    def load_saved_model(self, classifier_key, params, suffix=''):
        base_filename = self.filename.split('/')[-1].split('.')[0]

        tail_string = self.get_tail_string(params)
        model_name = '%s_%s'%(classifier_key, tail_string)

        model_filename = 'models/'+base_filename+'_'+model_name+suffix

        weights = np.array([*self.datasets[classifier_key+'_weight']])
        loss = weightedLoss(tf.keras.losses.categorical_crossentropy, weights)
        self.models[model_name] = tf.keras.models.load_model(model_filename)#, custom_objects={"_my_ragged_tensor_lossFunc":loss})
        print('Loaded %s from %s'%(model_name, model_filename))

    def calc_top_mass_with_triplet(self, 
            classifier_key,
            params):



        X_test, y_test = self.datasets[classifier_key + '_test']

        tail_string = self.get_tail_string(params)
        classifier_name = '%s_%s'%(classifier_key, tail_string)
        classifier = self.models[classifier_name]

        yhat_test = classifier(X_test).numpy()

        masses = calc_top_mass(self.events_test, yhat_test)
        masses_true = calc_top_mass(self.events_test, y_test.numpy())

        return masses, masses_true


    def calc_W_mass_with_doublet(self,
            classifier_key,
            params):

        X_test, y_test = self.datasets[classifier_key + '_test']

        tail_string = self.get_tail_string(params)
        classifier_name = '%s_%s'%(classifier_key, tail_string)
        classifier = self.models[classifier_name]

        yhat_test = classifier(X_test).numpy()

        masses = calc_W_mass(self.events_test, yhat_test)
        masses_true = calc_W_mass(self.events_test, y_test.numpy())


        return masses, masses_true

    def get_test_dataset(self, classifier_key):
        return self.datasets[classifier_key + '_test']

    
    def load_model_weights(self, model_name, classifier_key, suffix=''):
        base_filename = self.filename.split('/')[-1].split('.')[0]
        model_filename = './models/'+base_filename+'_'+model_name+suffix
        model = self.classifiers[classifier_key](**model_params_dict[classifier_key])
        model.load_weights(model_filename+'/variables/variables')
        return model

    def load_model_special(self, classifier_key, params, suffix=''):
        '''Special function to load model cause tensorflow hates ragged tensors'''
        tail_string = self.get_tail_string(params)
        model_name = '%s_%s'%(classifier_key, tail_string)
        return self.load_model_weights(model_name, classifier_key, suffix=suffix)

