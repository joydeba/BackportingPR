# BackportingPR
The tool ReBack can recommend backporting pull-requests in other versions.
Reback works in three moduls, Preprocessing, Embedding, and Classification. Below Figure shows module and necessary inputs. 

![ProposedApproachStablePR](https://user-images.githubusercontent.com/2823041/129495580-6244b3d6-3b52-4848-b0e4-395a0c98025b.png)

# Required libraries to install for the first time.

- Python 2.7.17 (https://docs.anaconda.com/anaconda/install/linux/)
  - conda create -n python2 python=2.7 anaconda
  - source activate python2   

- Tensorflow 1.10.0 (https://www.tensorflow.org)
  -  conda install -c conda-forge tensorflow=1.10.0

- Numpy 1.16.6 (https://www.numpy.org)
  -   pip install numpy==1.16.6

- Scikit-learn 0.20.4 (https://scikit-learn.org/stable/)
  -   pip install scikit-learn==0.20.4

- Keras 2.3.1 (https://keras.io)
  - pip install Keras==2.3.1  

# Avaiable Datasets

- https://doi.org/10.5281/zenodo.5196363 

# Training the model

  Run the following command to train the model: 

	$ python ReBack.py --train --data ReBackTrainingSample.out --model reback
	
  Run the following command to train the model with hyperparameters:

	$ python ReBack.py --train --data ReBackTrainingSample.out --model reback --embedding_dim 128  --filter_sizes "1,2" --num_filters 64

# Testing the model

  Run the following command to test the model: 

	$ python ReBack.py --predict --data ReBAckTestingSample.out --model reback


