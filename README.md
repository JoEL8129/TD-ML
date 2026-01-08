## TDML - TouchDesigner Machine Learning Toolkit 


A set of [TouchDesigner](https://derivative.ca/) components integrating machine-learning algorithms into TD's node-based CHOP workflows. Based on [scikit-learn](https://scikit-learn.org), [skorch](https://skorch.readthedocs.io/en/stable/), [umap-learn](https://umap-learn.readthedocs.io/en/latest/). Inspired by [FluCoMa](https://www.flucoma.org/) for MaxMSP. 


![Project Status](https://img.shields.io/badge/Status-Beta-brightgreen)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS-blue)
![TouchDesigner](https://img.shields.io/badge/TouchDesigner-2023.12370%2B-orange)
![Python](https://img.shields.io/badge/Python-3.11-blue)

---
<br>

![img](./data/img/audio%20classification.png)
*Audio (Voice) Classification*
<br>


![img](./data/img/gesture%20classification.png)
*Facial Expression Classification (using face_tracking from [mediapipe](https://github.com/torinmb/mediapipe-touchdesigner) by Torin Blankensmith)*
<br>


![img](./data/img/gesture%20mapping.png)
![img](./data/img/gesture%20mapping%202.png)
*Hand Gesture Mapping to Drive Visual Synth based on Training Examples (Presets) (using hand_tracking from [mediapipe](https://github.com/torinmb/mediapipe-touchdesigner) by Torin Blankensmith)*

## Installation

You need to load a virtual python environment into TD:

### TouchDesigner 2025

The [TDPyEnvManager](https://derivative.ca/community-post/introducing-touchdesigner-python-environment-manager-tdpyenvmanager/72024) from Palette does that. 

1. Download [miniconda](https://www.anaconda.com/download/success) and install
2. Open TD and save the project somewhere.
3. Drop the [environment.yml](environment.yml) next to the project.
4. Back in TD, Drag and Drop [TDPyEnvManager](https://derivative.ca/community-post/introducing-touchdesigner-python-environment-manager-tdpyenvmanager/72024) from Palette to your Project
5. Select Conda Mode
6. Set the miniconda install path (point it to the miniconda install folder)
7. Toggle "Create from environment.yml" par, wait
8. When done, The env par should show 'td-ml' 
9. Set the 'activate' toggle parameter.

*This has to be done only once as the 'td-ml' env is now on your system in the miniconda folder, for loading it into other project you can just use the [TDPyEnvManager](https://derivative.ca/community-post/introducing-touchdesigner-python-environment-manager-tdpyenvmanager/72024) without the install part and just set your miniconda install path and the env name par to 'td-ml'*  

### TouchDesigner 2023

As this [tutorial](https://derivative.ca/community-post/tutorial/anaconda-miniconda-managing-python-environments-and-3rd-party-libraries) shows in more depth, [miniconda](https://www.anaconda.com/download/success) is a one way to include external python packages into touchdesigner projects. Once miniconda is downloaded and installed with recommended settings - an anaconda prompt window can be opened to type a few commands:

`conda create -n td-ml python=3.11 -y`

`conda activate td-ml`

`cd C:\Users\Username\Documents\GitHub\TD-ML` 
*(cd to downloaded TD-ML folder on your system, as requirements.txt lives there)*

`pip install -r requirements.txt`

Once that is done, open TouchDesigner and drop the [conda_env_loader.tox](main/tools/other/conda_env_loader.tox) and set miniconda installation folder path and env name and hit 'load'. (TD Restart might be needed)



## Components


### Neural Network

- **MLP-Classifier**: A multi-layer perceptron neural network for
classification tasks. 

- **MLP-Regressor**: A multi-layer perceptron neural network for
regression tasks. 

- **LSTM-Classifier**: A Long-Short-Term-Memory neural network for classifications tasks.

- **LSTM-Regressor**: A Long-Short-Term-Memory neural network for regression tasks.


### Manifold

- **UMAP** (Uniform Manifold Approximation and Projection): 
A dimensionality reduction tool that seeks to preserve both global and local structure of high-dimensional data when embedding it into a lower-dimensional space. 

### Clustering

- **K-Means**: A clustering component for unsupervised learning,
which groups similar data points into clusters.

### Utility


- **Datasetter**: Utility tool for the creation, editing and management of datasets from CHOP channels and DATs. 

- **Conda-Environment-Loader**: Quickly load a predefined Conda environment from within TouchDesigner. (not needed in TD2025)

- **Filter Samples**: Filters/Smooths across the samples of a CHOP channel using various filters from scipy.signal

