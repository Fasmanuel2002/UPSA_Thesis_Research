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
