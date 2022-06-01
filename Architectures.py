import tensorflow as tf
import tensorflow.keras.backend as K

import numpy as np

class MyActivation(tf.keras.layers.Layer):
    """custom activation to pass mask"""
    def __init__(self, activation):
        super(MyActivation, self).__init__()
        self.activation = activation
        self.supports_masking = True

    def call(self, inputs, mask=None):
        return self.activation(inputs)

class adder(tf.keras.layers.Layer):
    """custom adder function to maintain mask"""
    def __init__(self, mean=False):
        super(adder, self).__init__()
        self.mean = mean
        self.supports_masking = True

    def call(self, inputs, mask=None):  
        mask_expanded = tf.tile(tf.expand_dims(tf.cast(mask, 'float32'),-1), (1,1,tf.shape(inputs)[-1]))
        if(self.mean):
            return tf.reduce_sum(mask_expanded*inputs, axis= 1 )/tf.reduce_sum(mask_expanded, axis= 1 )

        return tf.reduce_sum(mask_expanded*inputs, axis= 1 )

    def compute_mask(self, inputs, mask=None):
        if mask is None:
            return None
        return None

class concat_special(tf.keras.layers.Layer):
    """custom layer to pool while maintaing mask"""
    def __init__(self):
        super(concat_special, self).__init__()
        self.supports_masking = True

    def call(self, inputs, mask=None):
        inputs[1] = tf.tile(tf.expand_dims(inputs[1], axis=1), [1,tf.shape(inputs[0])[1], 1])
        return tf.concat([inputs[0],inputs[1]], axis=2)


class pool_concat(tf.keras.layers.Layer):
    """custom layer to pool while maintaing mask"""
    def __init__(self, full=True):
        super(pool_concat, self).__init__()
        self.supports_masking = True
        self.full = full

    def call(self, inputs, mask=None):
        if(mask is not None):
            mean_pool = tf.math.reduce_mean(tf.ragged.boolean_mask(inputs, mask=mask), axis=1)
        else:
            mean_pool = tf.math.reduce_mean(inputs, axis=1, keepdims=True)

        if(self.full is False):
            return mean_pool
        mean_pool = tf.tile(mean_pool, [1,tf.shape(inputs)[1], 1])
        x = tf.concat([inputs, mean_pool], axis=2)
        return x


class DeepSet(tf.keras.Model):
    """Creates basic Deep Set architecture"""
    def __init__(self, width, depth, latent_dim, Sigma=tf.nn.leaky_relu, final_Sigma=tf.nn.softmax, initial_mask=True, pooled=False, mean=False):
        super(DeepSet, self).__init__()
        depth = int(depth)
        width = int(width)
        latent_dim = int(latent_dim)
        self.width = width
        self.depth = depth
        self.Sigma = MyActivation(Sigma)
        self.depth = depth
        self.final_Sigma = final_Sigma

        self.Phi = [tf.keras.layers.Dense(width) for _ in range(depth-1)]
        self.Phi.append(tf.keras.layers.Dense(latent_dim))

        self.Adder = adder(mean=mean)

        self.F = [tf.keras.layers.Dense(width) for _ in range(depth)]
        self.F.append(tf.keras.layers.Dense(2))
        self.initial_mask = initial_mask
        self.pooled = pooled

    def call(self, inputs):
        x = inputs

        #Apply Phi
        if(self.initial_mask==True):
            x = tf.keras.layers.Masking()(inputs)

        #if we have pooled version
        if(self.pooled):
            x = pool_concat()(x)

        for i in range(self.depth):
            x = self.Sigma(tf.keras.layers.TimeDistributed(self.Phi[i])(x))

        #Sum
        x = self.Adder(x)

        #Apply F
        for i in range(self.depth):
            x = self.Sigma.activation(self.F[i](x))

        #Softmax Activation for classification or some other final activation (sigma)
        x = self.final_Sigma(self.F[-1](x))
        return x

class NestedConcat(tf.keras.Model):
    def __init__(self, width, depth, latent_dim, L=1, Sigma=tf.nn.leaky_relu, final_Sigma=tf.nn.softmax, initial_mask=True, pooled=False, mean=False):
        super(NestedConcat, self).__init__()
        depth = int(depth)
        width = int(width)
        latent_dim = int(latent_dim)

        self.width = width
        self.depth = depth
        self.Sigma = MyActivation(Sigma)
        self.depth = depth
        self.final_Sigma = final_Sigma
        self.N = L

        self.Phi = [[tf.keras.layers.Dense(width) for _ in range(depth-1)] for i in range(L)]
        self.Phi[0].append(tf.keras.layers.Dense(latent_dim))


        self.Adder = adder(mean=mean)

        self.F = [[tf.keras.layers.Dense(width) for _ in range(depth)] for i in range(L)]
        self.F_final = tf.keras.layers.Dense(2)
        self.initial_mask = initial_mask

    def call(self, inputs):

        x = tf.keras.layers.Masking()(inputs)

        xhat = None
        for i in range(1,self.N+1):
            c_Phi = self.Phi[-i]
            c_F   = self.F[-i]
            if(xhat != None):
                xhat = concat_special()([x,xhat])
            else:
                xhat = tf.keras.layers.Masking()(inputs)

            for layer in c_Phi:
                xhat = self.Sigma(tf.keras.layers.TimeDistributed(layer)(xhat))

            xhat = self.Adder(xhat)
            for layer in c_F:
                xhat = self.Sigma.activation(layer(xhat))

        return self.final_Sigma(self.F_final(xhat))

class DeepSetLayer(tf.keras.layers.Layer):
    """Implements Lambda-Gamma Pooling version 
    of equivariant deep sets layer"""
    def __init__(self, in_features, out_features, n):
        super(DeepSetLayer, self).__init__()

        self.supports_masking = True
        self.out_features = out_features
        self.in_features = in_features

        #Inputs to layes will be in_features length mathbfors
        #with variable length timestamps
        #e.g. inputs shape = (batch_size, (variable # of jets), in_features)
        #and output shape = (batch_size, (same variable # of jets), out_features)

        self.Gamma = self.add_weight(name='Gamma'+str(n),
                shape=(in_features, out_features), 
                initializer='he_uniform',
                trainable=True)

        self.Lambda = self.add_weight(name='Lambda'+str(n),
                shape=(in_features, out_features), 
                initializer='he_uniform', 
                trainable=True)
        
    def call(self, inputs, mask=None):
        #uses ragged map_falt_values for Ragged tensors to handle
        #variable number of jet
        xG = tf.matmul(inputs, self.Gamma)
        mean_pool = tf.math.reduce_mean(tf.ragged.boolean_mask(inputs, mask=mask), axis=1, keepdims=True)
        xL = tf.matmul(mean_pool, self.Lambda)
        ones = tf.ones([tf.shape(inputs)[1], 1])
        xL = tf.matmul(ones, xL)
        assert(xG.shape[-1] == self.out_features)
        assert(xL.shape[-1] == self.out_features)
        return (xG - xL)

class DeepSetEquivariantTransform(tf.keras.Model):
    """Implements Deep Set with Equivariant Transform"""
    def __init__(self, in_features,width, depth, latent_dim, Sigma=tf.nn.leaky_relu, final_Sigma=tf.nn.softmax, initial_mask=True):
        super(DeepSetEquivariantTransform, self).__init__()
        self.width = width
        self.depth = depth
        self.in_features = in_features
        self.latent_dim = latent_dim

        self.Gk = [DeepSetLayer(self.in_features, self.width, 0)]
        for i in range(depth-2):
            self.Gk.append(DeepSetLayer(self.width, self.width, i+1))
        self.Gk.append(DeepSetLayer(self.width, self.latent_dim, 100))
        self.Sigma = MyActivation(Sigma)
        self.final_Sigma = MyActivation(final_Sigma)

        self.Adder = adder()

        self.F = [tf.keras.layers.Dense(width) for _ in range(depth)]
        self.F.append(tf.keras.layers.Dense(2))
        self.initial_mask = initial_mask

    def call(self, inputs):
        x = inputs

        if(self.initial_mask==True):
            x = tf.keras.layers.Masking()(inputs)

        for layer in self.Gk:
            x = self.Sigma(layer(x))

        x = self.Adder(x)

        #Apply F
        for i in range(self.depth):
          x = self.Sigma.activation(self.F[i](x))

        #Softmax Activation for classification or some other final activation (sigma)
        x = self.final_Sigma(self.F[-1](x))
        return x

def adj(num_points, features, idxs=None):
    queries_shape = tf.shape(features)
    batch_size = queries_shape[0]

    idxs = tf.tile(tf.expand_dims(idxs,axis=0), [batch_size,1,1])

    batch_indices = tf.tile(tf.reshape(tf.range(batch_size), (-1, 1, 1, 1)), (1, num_points, num_points, 1))
    indices = tf.concat([batch_indices, tf.expand_dims(idxs, axis=3)], axis=3)  # (N, P, K, 2)
    return tf.gather_nd(features, indices)

def all_but(idx, n_jets):
    ret = [i for i in range(n_jets)]
    del ret[idx]
    return ret

class EdgeConvLayer(tf.keras.layers.Layer):
    '''Implements something similar to 
    https://github.com/hqucms/ParticleNet/blob/master/tf-keras/tf_keras_model.py'''
    def __init__(self, widths, num_particles, depth=3, centered=False, shortcut=False):
        super(EdgeConvLayer, self).__init__()
        depth = int(depth)
        self.supports_masking = True
        self.num_particles = num_particles
        self.K = self.num_particles
        self.depth = len(widths)
        self.widths = widths
        self.centered = centered
        self.supports_masking = True

        self.idxs = [[j for j in range(num_particles)] for i in range(num_particles)]

        self.linears = [tf.keras.layers.Conv2D(widths[i], kernel_size=1, data_format='channels_last',
                               use_bias=True, kernel_initializer='glorot_normal', activation=MyActivation(tf.nn.leaky_relu)) for i in range(depth)]
       


    def call(self, features, mask=None):
        fts = features

        adj_fts = adj(self.num_particles, fts, idxs = self.idxs)
        adj_fts_center = tf.tile(tf.expand_dims(fts, axis=2), (1, 1, self.K, 1))
        
        if(self.centered):
            adj_fts = tf.subtract(adj_fts, adj_fts_center)
            
        adj_fts = tf.concat([adj_fts_center, adj_fts], axis=-1)
        x = adj_fts
        for idx in range(self.depth):
            x = self.linears[idx](x)

        mask_expanded = tf.tile(tf.expand_dims(tf.cast(mask, 'float32'),-1), (1,1,tf.shape(x)[-1]))
        mask_fts = adj(self.num_particles, mask_expanded, idxs = self.idxs)
        mask_fts_centered = tf.tile(tf.expand_dims(mask_expanded, axis=2), (1, 1, self.K, 1))        
        mask_fts = mask_fts * mask_fts_centered
        
        fts = tf.math.multiply_no_nan(1/tf.reduce_sum(mask_fts, axis=2), tf.reduce_sum(mask_fts*x, axis= 2))
        ret = fts
        
        return tf.keras.layers.Activation(tf.nn.leaky_relu)(ret)

    def compute_mask(self, inputs, mask=None):
        if mask is None:
            return None
        return mask

class Pairwise(tf.keras.Model):
    """constructs edgeconv sequentially combined with deep set"""
    def __init__(self, depth, ec_widths, width, Sigma=tf.nn.leaky_relu, final_Sigma=tf.nn.softmax, initial_mask=True, num_particles=15, centered=False, shortcut=False):
        super(Pairwise, self).__init__()
        self.depth = depth
        self.edge_convs = EdgeConvLayer(ec_widths, num_particles, depth=len(ec_widths), centered=centered, shortcut=shortcut)

        self.Sigma = MyActivation(Sigma)
        self.final_Sigma = final_Sigma

        self.Adder = adder()

        self.F = [tf.keras.layers.Dense(width) for _ in range(depth)]
        self.F.append(tf.keras.layers.Dense(2))
        self.initial_mask = initial_mask

    def call(self, inputs):
        x = inputs
        if(self.initial_mask==True):
            x = tf.keras.layers.Masking()(inputs)
        x = self.edge_convs(x)
        x = self.Adder(x)
       #Apply F
        for i in range(self.depth):
            x = self.Sigma.activation(self.F[i](x))

        #Softmax Activation for classification or some other final activation (sigma)
        x = self.final_Sigma(self.F[-1](x))
        return x

class IteratedPiPairwise(tf.keras.Model):
    def __init__(self, depth, ec_widths, width, latent_dim, Sigma=tf.nn.leaky_relu, final_Sigma=tf.nn.softmax, initial_mask=True, num_particles=15, centered=False, shortcut=False):
        super(IteratedPiPairwise, self).__init__()
        self.depth = depth
        
        if(len(np.array(ec_widths).shape) != 2):
            ec_widths = np.array([ec_widths])
        N = len(ec_widths)
        
        self.edge_convs = [EdgeConvLayer(ec_width, num_particles, depth=len(ec_width), centered=centered, shortcut=shortcut) for ec_width in ec_widths]
        self.Phi = [[tf.keras.layers.Dense(width, activation=Sigma) for _ in range(3)] for idx in range(N)]
        self.Phi[-1].append(tf.keras.layers.Dense(latent_dim))
 
        self.final_Sigma = final_Sigma

        self.Adder = adder()

        self.F = [tf.keras.layers.Dense(width, activation=Sigma) for _ in range(depth)]
        self.F.append(tf.keras.layers.Dense(2))
        self.initial_mask = initial_mask

    def call(self, inputs):
        x = inputs
        if(self.initial_mask==True):
            x = tf.keras.layers.Masking()(inputs)

        for idx in range(len(self.Phi)):
            x = self.edge_convs[idx](x)
    
            for layer in self.Phi[idx]:
                x = tf.keras.layers.TimeDistributed(layer)(x)

        x = self.Adder(x)

       #Apply F
        for i in range(self.depth):
            x = self.F[i](x)

        #Softmax Activation for classification or some other final activation (sigma)
        x = self.final_Sigma(self.F[-1](x))
        return x


class ExtendedEdgeConvLayer(tf.keras.layers.Layer):
    '''Implements something similar to 
    https://github.com/hqucms/ParticleNet/blob/master/tf-keras/tf_keras_model.py'''
    def __init__(self, widths, num_particles, depth=3, centered=False, shortcut=False):
        super(ExtendedEdgeConvLayer, self).__init__()
        depth = int(depth)
        self.supports_masking = True
        self.num_particles = num_particles
        self.K = self.num_particles
        self.depth = depth
        self.widths = widths
        self.idxs = [[j for j in range(num_particles)] for i in range(num_particles)]

        self.linears = [tf.keras.layers.Conv3D(widths[i], kernel_size=(1, 1, 1), strides=1, data_format='channels_last',
                               use_bias=True, kernel_initializer='glorot_normal', activation=MyActivation(tf.nn.leaky_relu)) for i in range(depth)]
        
    def call(self, features, mask=None):
        fts = features

        #first we create template for one layer
        adj_fts = adj(self.num_particles, fts, idxs = self.idxs)
        adj_fts_center = tf.tile(tf.expand_dims(fts, axis=2), (1, 1, self.K, 1))
        
        adj_fts = tf.concat([adj_fts_center, adj_fts], axis=-1)
        
        #now we build whole cube
        adj_fts = tf.tile(tf.expand_dims(adj_fts, axis=1), (1, self.K, 1, 1, 1))
        adj_fts_center = tf.tile(tf.expand_dims(tf.expand_dims(fts, axis=2), axis=2), (1, 1, self.K, self.K, 1))
        x = tf.concat([adj_fts_center, adj_fts], axis=-1)
                
        for idx in range(self.depth):
            x = self.linears[idx](x)

        mask_expanded = tf.tile(tf.expand_dims(tf.cast(mask, 'float32'),-1), (1,1,tf.shape(x)[-1]))
        mask_fts = adj(self.num_particles, mask_expanded, idxs = self.idxs)
        mask_fts_centered = tf.tile(tf.expand_dims(mask_expanded, axis=2), (1, 1, self.K, 1))     
        
        mask_fts = tf.tile(tf.expand_dims(mask_fts, axis=1), (1, self.K, 1, 1, 1))
        mask_fts_centered = tf.tile(tf.expand_dims(tf.expand_dims(mask_expanded, axis=2), axis=2), (1, 1, self.K, self.K, 1))

        mask_fts = mask_fts * mask_fts_centered            
            
        x = tf.reduce_sum(mask_fts*x, axis= 3)/ tf.reduce_sum(mask_fts, axis=3)
        x = tf.math.reduce_mean(x, axis=2)
        
        return tf.keras.layers.Activation(tf.nn.leaky_relu)(x)
    
    
class Tripletwise(tf.keras.Model):
    def __init__(self, depth, ec_widths, width, Sigma=tf.nn.leaky_relu, final_Sigma=tf.nn.softmax, initial_mask=True, num_particles=15, centered=False, shortcut=False):
        super(Tripletwise, self).__init__()
        self.depth = depth
        self.edge_convs = ExtendedEdgeConvLayer(ec_widths, num_particles, depth=len(ec_widths), centered=centered, shortcut=shortcut)

        self.Sigma = MyActivation(Sigma)
        self.final_Sigma = final_Sigma

        self.Adder = adder()

        self.F = [tf.keras.layers.Dense(width) for _ in range(depth)]
        self.F.append(tf.keras.layers.Dense(2))
        self.initial_mask = initial_mask

    def call(self, inputs):
        x = inputs
        if(self.initial_mask==True):
            x = tf.keras.layers.Masking()(inputs)

        x = self.edge_convs(x)
        x = self.Adder(x)

       #Apply F
        for i in range(self.depth):
            x = self.Sigma.activation(self.F[i](x))

        #Softmax Activation for classification or some other final activation (sigma)
        x = self.final_Sigma(self.F[-1](x))
        return x

class DNN_Classifier(tf.keras.Model):
    """simple dense neural netowkr classifier"""
    def __init__(self, width, depth, num_classes):
        super(DNN_Classifier, self).__init__()
        self.width = width
        self.depth = depth
        self.num_classes = num_classes
        self.model = [tf.keras.layers.Dense(self.width, activation=tf.nn.leaky_relu) for i in range(self.depth)]
        self.model.append(tf.keras.layers.Dense(self.num_classes, activation='softmax'))
        self.batch_norm = tf.keras.layers.BatchNormalization()

    def call(self, inputs):
        x = self.batch_norm(inputs)
        for layer in self.model:
            x = layer(x)
        return x


class DNN_Flatten(tf.keras.Model):
    """simple dense neural netowkr classifier"""
    def __init__(self, width, depth, num_classes):
        super(DNN_Flatten, self).__init__()
        self.width = width
        self.depth = depth
        self.num_classes = num_classes
        self.model = [tf.keras.layers.Dense(self.width, activation=tf.nn.leaky_relu) for i in range(self.depth)]
        self.model.append(tf.keras.layers.Dense(self.num_classes, activation='softmax'))
        self.batch_norm = tf.keras.layers.BatchNormalization()

    def call(self, inputs):
        x = tf.keras.layers.Flatten()(inputs)
        x = self.batch_norm(x)
        for layer in self.model:
            x = layer(x)
        return x
    
class RaggedGetter(tf.keras.layers.Layer):
    def call(self, inputs, mask=None):
        return tf.ragged.boolean_mask(inputs, mask=mask)


class LatentGetter(tf.keras.Model):
    """helper class to extract the latent representations of events"""
    def __init__(self, submodel, condensed=False):
        super(LatentGetter, self).__init__()
        self.submodel = submodel
        self.condensed = condensed

    def call(self, inputs):
        x = tf.keras.layers.Masking()(inputs)
        for layer in self.submodel:
            x = layer(x)
        if(not self.condensed):
            x = RaggedGetter()(x)
        return x

    
    
    
model_params_dict = {'particlewise':{'width':128, 'depth':4, 'latent_dim':64}, 
                     'particlewise_mean':{'width':128, 'depth':4, 'latent_dim':64,'mean':True}, 
        'nested_concat':{'width':70, 'depth':4, 'latent_dim':64, 'L':3},
        'pairwise': {'depth':5, 'ec_widths':(64,128,256,128,64), 'width':64},
        'pairwise_nl': {'depth':5, 'ec_widths':((64,128,256,128,64)), 'width':32, 'latent_dim':64},
        'pairwise_nl_iter': {'depth':5, 'ec_widths':((64,64,116,64,64),(64,64,116,64,64),(64,64,116,64,64)), 'width':32, 'latent_dim':64},
        'tripletwise': {'depth':5, 'ec_widths':(64,128,256,128,64), 'width':64},
        'dnn': {'width':256,'depth':3,'num_classes':2},
        'naivednn':{'width':256,'depth':3,'num_classes':2}
        }
classifiers_name = {'particlewise':r'Particlewise', 
                    'nested_concat':r'Nested Concatenation',
                    'pairwise':r'Pairwise', 
                    'tripletwise':r'Tripletwise', 
                    'pairwise_nl':r'Nonlinear Pairwise',
                    'pairwise_nl_iter':r'Iterated Nonlinear Pairwise' ,
                    'dnn':'dNN + ATLAS Features', 
                    'naivednn':'dNN + Naive Features'}

classifiers = {'particlewise':DeepSet, 
               'particlewise_mean':DeepSet,
            'nested_concat':NestedConcat,
            'pairwise':Pairwise,
            'tripletwise':Tripletwise,
            'dnn':DNN_Classifier,
            'naivednn':DNN_Flatten,
            'pairwise_nl':IteratedPiPairwise,
            'pairwise_nl_iter':IteratedPiPairwise}

