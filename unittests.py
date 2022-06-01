import unittest

import sys
sys.path.insert(1, '../')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np

from random import random, shuffle
from Architectures import *

def test_padding(model, model2=None):
    working = True
    for tests in range(10):
        event = [[random() for i in range(7)] for j in range(5)]
        event2 = np.array(event.copy())
        event2 = np.concatenate([event2, np.zeros((20, 7))], axis=0)

        data1 = tf.constant([event])
        data2 = tf.constant([event2])

        yhat1 = model.predict(data1)
        yhat2 = None
        if(model2 is None):
            yhat2 = model.predict(data2)
        else:
            yhat2 = model2.predict(data2)
            model2.set_weights(model.get_weights())
            yhat2 = model2.predict(data2)


        
        for i in range(2):
            if(abs(yhat1[0][i] - yhat2[0][i])>1e-5 or np.isnan(yhat1[0][i]) or np.isnan(yhat2[0][i])):
                print(yhat1[0][i],  yhat2[0][i])
                working=False
    return working


def test_permute(model):
    working = True
    for tests in range(10):
        event = [[random() for i in range(7)] for j in range(5)]
        event2 = event.copy()
        shuffle(event2)

        data = tf.constant([event, event2])

        yhat = model.predict(data)
        for i in range(2):
            if(abs(yhat[0][i] - yhat[1][i]) > 1e-5):
                print('permute', abs(yhat[0][i] - yhat[1][i]))
                working = False
    return working
         
class test_particlewise_mean(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(test_particlewise_mean, self).__init__(*args, **kwargs)
        name = 'particlewise'
        self.model = classifiers[name](**model_params_dict[name], mean=True)

    def test_padding(self):
        self.assertEqual(test_padding(self.model), True)

    def test_permutation(self):
        self.assertEqual(test_permute(self.model), True)


class test_particlewise(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(test_particlewise, self).__init__(*args, **kwargs)
        name = 'particlewise'
        self.model = classifiers[name](**model_params_dict[name])

    def test_padding(self):
        self.assertEqual(test_padding(self.model), True)

    def test_permutation(self):
        self.assertEqual(test_permute(self.model), True)

class test_nested_concat(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(test_nested_concat, self).__init__(*args, **kwargs)
        name = 'nested_concat'
        self.model = classifiers[name](**model_params_dict[name])

    def test_padding(self):
        self.assertEqual(test_padding(self.model), True)

    def test_permutation(self):
        self.assertEqual(test_permute(self.model), True)

class test_pairwise(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(test_pairwise, self).__init__(*args, **kwargs)
        name = 'pairwise'
        self.name = name
        self.model = classifiers[name](**model_params_dict[name], num_particles=5)
        self.model2 = classifiers[name](**model_params_dict[name], num_particles=25)

    def test_padding(self):
        self.assertEqual(test_padding(self.model, model2=self.model2), True)

    def test_permutation(self):
        self.assertEqual(test_permute(self.model), True)



class test_pairwise_nl(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(test_pairwise_nl, self).__init__(*args, **kwargs)
        name = 'pairwise_nl'
        self.name = name
        self.model = classifiers[name](**model_params_dict[name], num_particles=5)
        self.model2 = classifiers[name](**model_params_dict[name], num_particles=25)

    def test_padding(self):
        self.assertEqual(test_padding(self.model, model2=self.model2), True)

    def test_permutation(self):
        self.assertEqual(test_permute(self.model), True)


class test_pairwise_nl_iter(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(test_pairwise_nl_iter, self).__init__(*args, **kwargs)
        name = 'pairwise_nl_iter'
        self.name = name
        self.model = classifiers[name](**model_params_dict[name], num_particles=5)
        self.model2 = classifiers[name](**model_params_dict[name], num_particles=25)

    def test_padding(self):
        self.assertEqual(test_padding(self.model, model2=self.model2), True)

    def test_permutation(self):
        self.assertEqual(test_permute(self.model), True)




class test_tripletwise(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(test_tripletwise, self).__init__(*args, **kwargs)
        name = 'tripletwise'
        self.name = name
        self.model = classifiers[name](**model_params_dict[name], num_particles=5)
        self.model2 = classifiers[name](**model_params_dict[name], num_particles=25)

    def test_padding(self):
        self.assertEqual(test_padding(self.model, model2=self.model2), True)

    def test_permutation(self):
        self.assertEqual(test_permute(self.model), True)


if __name__ == '__main__':
    unittest.main()
