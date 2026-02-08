# TD-ML - TouchDesigner Machine Learning Toolkit 


A set of [TouchDesigner](https://derivative.ca/) components integrating machine-learning algorithms into TD's node-based CHOP workflows. Based on [scikit-learn](https://scikit-learn.org), [PyTorch](https://pytorch.org/), [skorch](https://skorch.readthedocs.io/en/stable/), [umap-learn](https://umap-learn.readthedocs.io/en/latest/) and numerous other [packages](./requirements.txt). Inspired by [FluCoMa](https://www.flucoma.org/) for MaxMSP.
There is a [Discord](https://discord.gg/KzpH76P68X) for Questions, Sharing Datasets, Trained Models, Bug-Reports or Feedback. Video Tutorials will come soon, here on [YouTube](https://www.youtube.com/@bi.os_td).


![Project Status](https://img.shields.io/badge/Status-Beta-brightgreen)
![Platform](https://img.shields.io/badge/Platform-Windows-green)
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

You need to load a virtual python environment into TD based on provided [requirements.txt](./requirements.txt) or [environment.yml](./environment.yml):

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
9. Set the 'activate' toggle parameter. (might need a reload or restart of TD)

*This has to be done only once as the 'td-ml' env is now on your system in the miniconda folder, for loading it into other project you can just use the [TDPyEnvManager](https://derivative.ca/community-post/introducing-touchdesigner-python-environment-manager-tdpyenvmanager/72024) without the install part and just set your miniconda install path and select 'td-ml' from the found env names*  

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

- **MLP-Classifier**: A multi-layer perceptron neural network for classification tasks. A feedforward network that learns to map input features to discrete class labels, suitable for pattern recognition and categorical prediction tasks.

- **MLP-Regressor**: A multi-layer perceptron neural network for regression tasks. A feedforward network that learns to map input features to continuous numerical values, suitable for predicting continuous outputs and value estimation.

- **LSTM-Classifier**: A Long-Short-Term-Memory neural network for classification tasks. Designed to handle sequential data with temporal dependencies, making it ideal for time-series classification, gesture recognition, and other sequence-based categorical predictions. *(Work in progress)*

- **LSTM-Regressor**: A Long-Short-Term-Memory neural network for regression tasks. Designed to handle sequential data with temporal dependencies, making it ideal for time-series forecasting, continuous sequence prediction, and other regression tasks involving temporal patterns. *(Work in progress)*


### Manifold

- **UMAP** (Uniform Manifold Approximation and Projection): 
A dimensionality reduction tool that seeks to preserve both global and local structure of high-dimensional data when embedding it into a lower-dimensional space.

- **Isomap**: Isometric Mapping - a nonlinear dimensionality reduction method that preserves geodesic distances between data points. *(Work in progress - example implementations)*

- **SpectralEmbedding**: Spectral Embedding - a nonlinear dimensionality reduction method based on the spectral decomposition of the graph Laplacian. *(Work in progress - example implementations)*

### Decomposition 

- **PCA**: Principal Component Analysis - a linear dimensionality reduction technique that projects data onto lower-dimensional subspaces while preserving maximum variance. *(Work in progress - example implementations)*

- **KernelPCA**: Kernel Principal Component Analysis - a nonlinear extension of PCA that uses kernel functions to map data to higher dimensions before applying PCA. *(Work in progress - example implementations)*

- **SparsePCA**: Sparse Principal Component Analysis - a variant of PCA that produces sparse components, useful for interpretability and feature selection. *(Work in progress - example implementations)*

### Clustering

- **K-Means**: A clustering component for unsupervised learning,
which groups similar data points into clusters. *(Work in progress)*

### Helpers

- **Datasetter**: Comprehensive dataset management tool for creating, editing, and managing datasets from CHOP channels and DATs. Features include recording and snapping data, save/load functionality, data sanitization (filling empty cells, normalization, standardization)*(Work in progress)*, outlier detection and handling, missing value removal, and weight assignment based on data quality conditions. For training sequential neural networks there is an option to add seq_id. 

- **Conda-Environment-Loader**: Quickly load a predefined Conda environment from within TouchDesigner. (not needed in TD2025)

- **Filter Samples**: Filters/Smooths across the samples of a CHOP channel using various filters from scipy.signal

- **Dynamic Inputs**: Example tox illustrating dynamic type of COMPs input switching, limited to 3 at the moment. *(Work in progress - example implementations)*

- **Visual Synth**: Noisy Pattern Visuals Generator, used as Example for illustrating use-cases.

- **Wavetable Synth**: Wave Table Audio Synth, used as Example for illustrating use-cases.

### Audio (not included)

- **[td_mfcc](https://github.com/hrtlacek/td_mfcc )** by hrtlacek / Patrik Lechner


