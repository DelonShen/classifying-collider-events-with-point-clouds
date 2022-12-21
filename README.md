# Classifying Collider Events with Point Clouds

Implementation of architectures in [Comparing Point Cloud Strategies for Collider Event Classification](https://arxiv.org/abs/xxxx.xxxxx)

------
*If you have any questions at all about the code or using these kinds of architectures in your own analysis I'm always open to chatting, just let me know at [hi@delonshen.com](mailto:hi@delonshen.com)*

## Basics

To get started create a Anaconda environment with the provided `environment.yml` file 
```
conda env create -f environment.yml
conda activate pcec
python -m ipykernel install --user --name pcec --display-name "pcec"
```
Models are implemented with Keras/Tensorflow. An simple example of using one our pairwise architecture recommend in the paper for event classification is found in [`SimpleArchitectureDemo.ipynb`](SimpleArchitectureDemo.ipynb). Sample data files are stored in the `data/` folder. 

## Architectures
The architectures are implemented in `Architectures.py` as [`tf.keras.Model`](https://www.tensorflow.org/api_docs/python/tf/keras/Model) with the following names
- **Particlewise Architecture**: `DeepSet`. 
    - `width`: the number of nodes in the hidden layers of the neural networks parameterizing F and Φ
    - `depth`: the number of layers in each of the neural networks parameterizing F and Φ
    - `latent_dim`: the dimension of the latent space that particles are mapped to 
- **Pairwise/Tripletwise Architecture**: `Pairwise/Tripletwise`. 
    - `width`: the number of nodes in the hidden layers of the neural networks parameterizing F 
    - `depth`: the number of layers in each of the neural networks parameterizing F 
    - `ec_widths`: a list containing the number of nodes in each layer of Φ 
    - `num_particles`: the number of particles you pad your events to (see `N_PAD` below)
- **(Iterated) Nonlinear Pairwise Architecture**: `IteratedPiPairwise`: 
    - `depth`: the number of layers in each of the neural networks parameterizing F<sup>(i)</sup>
    - `width`: the number of nodes in the hidden layers of the neural networks parameterizing F<sup>(i)</sup>
    - `ec_widths`: a list of lists where each list contains the number of nodes in each layer of Φ<sup>(i)</sup>
    - `num_particles`: the number of jets you pad your events to (see `N_PAD` below)
- **Nested Concatenation Architecture**: `NestedConcat`:
    - `L`: number of nested Deep Sets to use. When `L=1` then `NestedConcat` is equivalent to `DeepSet`
    - `width`: the number of layers in each of the neural networks parameterizing F<sup>(i)</sup> and Φ<sup>(i)</sup>
    - `depth`: the number of layers in each of the neural networks parameterizing F<sup>(i)</sup> and Φ<sup>(i)</sup> (note: Φ<sup>(i)</sup> when `i≠L` have `depth-1` layers)
    - `latent_dim`: the dimension of the latent space that particles are mapped to 
    
    
You can get any of these models just by calling them:
```
from Architectures import *
classifier = DeepSet(depth=3, width=128, latent_dim=64)
```

These have all the methods defined in the tensorflow documentation for [`tf.keras.Model`](https://www.tensorflow.org/api_docs/python/tf/keras/Model), the most important of which are `fit` for training the model and `predict` for classifying events

Parameters used in paper are stored in dictionaries at the very bottom of the `Architectures.py` file. For example if you want to create the Pairwise architecture with the same parameter as used in the experiments in the paper you could run
```
from Architectures import *
classifier = classifiers['pairwise'](**model_params_dict['pairwise'])
```

## Data

Input data should have shape `(N_events, N_PAD, N_features)`. 
- `N_events` is the number of collider events in your dataset
- `N_PAD` is the number of particles you zero-pad each event. The zero padded particles are ignored by the architecture through masking layers but required for computationally efficient implementation. `N_PAD` should be larger than or equal to the maximum number of particles you have in one event in your dataset. If you have a dataset with shape `(N_events, -1, N_features)` where `-1` implies that each event has a variable number of particles you can call `tensorflow.keras.preprocessing.sequence.pad_sequences(events, dtype='float32', padding='post', maxlen=N_PAD)` to generate a padded version of your dataset suitable to be passed to the architectures.
- `N_features` is the number of features you describe each particle in your event with. We recommend mimicing the features chosen in the paper described in equation (14) (TODO check with final version)

## Generating Figures in Paper

The (very messy) code used to generate results in paper can be found in the directory `paper_code/`. The scripts to train the models is in `paper_code/gen_models.py` and the code to generate the figures in the paper are scattered throughout the other files. Namely:
- Fig 3 and Table 1 from `gen_plots.py`
- Table 3 from `latent_scan.py` and `latent_scan_aux.ipynb`
- Fig 4 and Fig 5 from `l2l8.ipynb`
- Fig 6 from `spearman_rank_analysis.ipynb`
- Fig 7 from `ditau_study.ipynb`

All this code and the `paper_code` folder itself is truly quite messy so if you have any questions about anything in this folder please don't hesitate to reach out at [hi@delonshen.com](mailto:hi@delonshen.com).  