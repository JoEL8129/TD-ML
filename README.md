# TD-ML

A TouchDesigner Machine Learning Toolkit. 

Aimed at easy accessibility and at a quick, but configurable way of applying machine learning directly within TouchDesigner on CHOPs & DATs, without the need to deal too much with data conversions & python. Wraps around scikit-learn, umap-learn and script chops.


## Installation 

The installation requires to enable external python packages to be importable into touchdesigner. 

As [this](https://derivative.ca/community-post/tutorial/anaconda-miniconda-managing-python-environments-and-3rd-party-libraries) tutorial shows in more depth, [miniconda](https://www.anaconda.com/download/success) is a good way to include external python packages into your touchdesigner projects. Once miniconda is downloaded and installed with recommended settings - you can open an anaconda prompt window and start typing commands. 

*First, create a new environment, it's called "td-ml" in this example (use any name you want), but make sure the python version matches that of your touchdesigner version (shown when opening a textport window in touch)*

*As of now (TouchDesigner 2023 / 2025) its 3.11*

`conda create -n td-ml python=3.11 -y`

`conda activate td-ml`

`cd C:\Users\Username\Documents\GitHub\TD-ML` (cd to downloaded TD-ML folder, as requirements.txt lives there)

`pip install -r requirements.txt`

Once that is done you can open TouchDesigner and if you're in Version 2023 drop the conda_env_loader.tox comp and put in your miniconda installation folder path and your env name. (Restart of project might be needed) 

If you're in Version 2025 just use the [TDPyEnvManager](https://derivative.ca/community-post/introducing-touchdesigner-python-environment-manager-tdpyenvmanager/72024), choose conda and select environment. Or use venv settings and choose install from requirements.txt.

