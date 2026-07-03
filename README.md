### A breast cancer investigation, developed in collaboration with the universitt
 
 <p align="center">
 <img width="900" height="350" alt="image" src="https://github.com/user-attachments/assets/c7958d86-afff-47ce-ba86-fa6fb24e15fc" />
 </p>

This research project is to compare five survival analysis models (penalized Cox
regression (Lasso, Elastic Net, and Ridge), Random Survival Forest, and two deep neural network
approaches (DeepSurv and Cox-nnet)) using RNA-seq data from 519 breast cancer patients from
TCGA, with overall survival administratively censored at 60 months.


## Installation Dependencies
The first step is to create the image from the dependencies to run this proyect
```bash
  docker compose up
```

Dependencies:

- [Pandas](https://pandas.pydata.org/): A data analysis and manipulation library used to work with structured datasets, tables,
and data frames.

- [NumPy](https://numpy.org/doc/stable/index.html): A numerical computing library that provides efficient multidimensional arrays and
mathematical operations for scientific computing.

- [InMoose](https://inmoose.readthedocs.io/en/latest/): A bioinformatics library for analyzing biology data, including gene expression and highthroughput biological datasets.

- [Limma](https://inmoose.readthedocs.io/en/stable/limma.html): A Bioconductor package widely used for differential gene expression analysis in microarray and RNA-sequencing studies.

- [Scikit-learn](https://scikit-learn.org/stable/) A machine learning library that provides algorithms for classification, regression, clustering, model evaluation, and data preprocessing.

- [Scikit-survival](https://scikit-survival.readthedocs.io/en/stable/): An extension of scikit-learn designed for survival analysis.

- [Pytorch](https://pytorch.org/): A deep learning framework used for building, training, and deploying neural networks and other machine learning models.

- [PyCox](https://github.com/havakv/pycox): A Python library for deep learning based survival analysis, built on top of PyTorch anddesigned for time-to-event prediction tasks.

- [SHAP](https://shap.readthedocs.io/en/latest/): An explainable AI library that interprets machine learning models by quantifying the contribution of each feature to a prediction.

 - [matplotlib](https://matplotlib.org/): Comprehensive library for creating static, animated, and interactive visualizations in Python

## Installation Dataset 
The second step consists of downloading the dataset used in this investigation from the CBioPortal

You need to visit the following website to download the dataset:
[Download from CBioPortal](https://www.cbioportal.org/study/summary?id=brca_tcga_pub2015)

Download the `Breast Invasive Carcinoma (TCGA, Cell 2015)` tar, which contains the dataset and put it on `data/`.

## Running the Project

- `Otto_TRACE/dataset/`  
  Contains the dataset processing pipeline based on the OTTO dataset. This class loads data from `train.jsonl`, constructs the model inputs, generates task-specific logits, and performs the input–target split used for training and evaluation.
  
- `Otto_TRACE/model/`
  
  This section presents the re-implementation of the TRACE model architecture, as described in Section 2.3 (Model Architecture) of the original TRACE paper, adapted for this research investigation
  
- `Otto_TRACE/training_models/`
  
  Contains the two versions of training pipeline for Single-Task Learning (STL) and Multi-Task Learning (MLT), the purpose of this code format is for jupyterhub GPU.

- `Otto_TRACE/test_models/`
  
  Contains the two versions of Testing Pipeline for Single-Task Learning (STL) and Multi-Task Learning (MLT).

- `Otto_TRACE/utils/`
  
  Contains six files for corresponding reasons
  - EarlyStopping: Stop training when a monitored metric F1 has stopped improving
  - feature_engineering: This file presents the re-implementation of the TRACE feature engineering, as described in Section 2.2 (Feature and Position Encoding) of the original TRACE paper
  - normalization: This file presents the re-implementation of TRACE for normalizating and log the time elapsed and time betwen from Section 2.2 (Feature and Position Encoding) of the original TRACE paper
  - plot_confussion_matrix: Script for Computing and Plotting the Confusion Matrix
  - SplitData: Script for splitting the dataset into training, validation, and test sets.
  - training_utils: Utility scripts designed to improve code readability, modularity, and maintainability.
