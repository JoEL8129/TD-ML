# TD-ML - TouchDesigner Machine Learning Toolkit 


A set of [TouchDesigner](https://derivative.ca/) components integrating machine-learning algorithms into TD's node-based CHOP workflows. Based on [scikit-learn](https://scikit-learn.org), [PyTorch](https://pytorch.org/), [skorch](https://skorch.readthedocs.io/en/stable/), [umap-learn](https://umap-learn.readthedocs.io/en/latest/) and numerous other [packages](./requirements.txt). Inspired by [FluCoMa](https://www.flucoma.org/) for MaxMSP.
There is a [Discord](https://discord.gg/KzpH76P68X) for Questions, Sharing Datasets, Trained Models, Bug-Reports or Feedback. Furthermore Tutorials will follow soon on [YouTube](https://www.youtube.com/@bi.os_td).


![Project Status](https://img.shields.io/badge/Status-Beta-brightgreen)
![Platform](https://img.shields.io/badge/Platform-Windows-green)
![TouchDesigner](https://img.shields.io/badge/TouchDesigner-2025.32280%2B-orange)
![Python](https://img.shields.io/badge/Python-3.11.10-blue)

---
<br>

![img](./data/img/1.gif)
*MLP Regressor: Slider2D Mapping to 28D Visual Synth Pars*

![img](./data/img/audio%20classification.png)
*MLP Classifier: Audio (Voice) Classification & UMAP Visualization*
<br>



## Description 

Started as learning/university project and it still is, but I'm trying to make it as generally usable as possible, but stuff might not be perfectly solved or 100% working yet and up for future changes, so better consider it Beta / Experimental. 

The idea is to train small neural networks from scratch on any CHOP data with a few Components and without (the need for) touching any python. Includes Classification & Regression Neural Networks (MLP & LSTM so far), some Dimension Reduction Tools like UMAP or tSNE and other things like a 'Datasetter' Tool for quickly creating/recording CHOP data into needed table formats / custom Datasets. Useful for any Classification or Regression Tasks on any CHOP data (tracking, sensors, audio, control parameters, etc.). Let me know (@discord or here) if you encounter issues. Hope you have fun with it ;)

## Installation

You need to load a virtual python environment into TD based on provided [requirements.txt](./requirements.txt) or [environment.yml](./environment.yml). Due to Torch (3gb) downloading can take a while (3.6gb in total):

### TouchDesigner 2025

The [TDPyEnvManager](https://derivative.ca/community-post/introducing-touchdesigner-python-environment-manager-tdpyenvmanager/72024) from Palette does that. 

#### Option I: venv 

1. Drag and Drop [TDPyEnvManager](https://derivative.ca/community-post/introducing-touchdesigner-python-environment-manager-tdpyenvmanager/72024) from Palette to your Project
2. Choose "venv" mode
3. Set the path to be the folder where you want to install the virtual environment to
4. Set the name it should have (eg. "td-ml")
5. Pulse "create venv from requirements.txt" (make sure you copied requirements.txt from the repo to your project folder)
6. Wait 

*In another project if you want to load that venv, you just have to set the path in the TDPyEnvManager correctly to where you installed it to the first time and set name accordingly*


#### Option II: conda 

1. `conda create -n td-ml python=3.11.10 -y`
2. `conda activate td-ml`
3. `cd C:\Users\Username\Documents\GitHub\TD-ML` 
*(cd to downloaded TD-ML folder on your system, as requirements.txt lives there)*
4. `pip install -r requirements.txt`
5. Drag and Drop [TDPyEnvManager](https://derivative.ca/community-post/introducing-touchdesigner-python-environment-manager-tdpyenvmanager/72024) from Palette to your Project
6. Choose "conda" mode
7. Set the path to be the folder where you want to installed conda to 
8. Select and Load 'td-ml'

*In another project if you want to load that conda venv, you just have to set the path in the TDPyEnvManager correctly to where you installed conda to and set name accordingly*

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


## More Usage Examples

![img](./data/img/3.gif)
*MLP Regressor: Leap Motion (1 Hand - 46 Channels) 46D Mapping to 28D Visual Synth Pars*

![img](./data/img/2.gif)
*MLP Regressor: Slider2D Mapping to 120D Wavetable Synth chop-samples)*
<br>

![img](./data/img/gesture%20classification.png)
*MLP Classifier: Facial Expression Classification (using face_tracking from [mediapipe](https://github.com/torinmb/mediapipe-touchdesigner) by Torin Blankensmith)*
<br>
