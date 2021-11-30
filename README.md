# EWS - Eschikon Wheat Segmentation



This repository contains code for reproducing benchmarks on the Eschikon Wheat Segmenation (EWS) Dataset.
Running the `main.py` trains, validates and tests the following methods:

- proper citation of Zenkl et al (TBD)
- Sadeghi-Tehran, P., Virlet, N., Sabermanesh, K., and Hawkesford, M. J. (2017).  Multi-feature machine learning model for automatic segmentation of green fractional vegetation cover for high-throughput field phenotyping.Plant methods13, 103
- Rico-Fernández, M., Rios-Cabrera, R., Castelan, M., Guerrero-Reyes, H.-I., and Juarez-Maldonado, A.(2019). A contextualized approach for segmentation of foliage in different crop species.Computers and Electronics in Agriculture156, 378–386
- Yu, K., Kirchgessner, N., Grieder, C., Walter, A., and Hund, A. (2017).  An image analysis pipeline for automated classification of imaging light conditions and for quantification of wheat canopy cover time series in field phenotyping. Plant Methods13, 1–13

## Installation

All the necessary dependencies can be installed into a new conda environment with `conda env create -f env.yml`. This command creates a new conda environment `EWS`.
This code utilizes Pytorch with GPU capabilities. Please check your cuda version. 

This repository utilizes code from `https://github.com/qubvel/segmentation_models.pytorch`

## Setup the Data

Download the dataset from: https://www.research-collection.ethz.ch/handle/20.500.11850/512332

Create following folder structure:
```
EWS
└── data
     └── train
     └── validation
     └── test
     └── Data for Yu et al
          └── train
          └── validation
          └── test
     
```

Or adjust the paths in `methods/base.py` and `methods/yu_et_al_2017.py`

## Quick GPU Test
If you are unsure if your setup can utilize a GPU, active the `EWS`environment and execute following snippet with python:

`
import torch
print(torch.zeros(1).cuda())
`

If no errors appear, you should be good to go. Otherwise, try reinstalling pytorch with different cuda version: https://pytorch.org/get-started/locally/

## Running 

In order to replicate the benchmarks, execute `main.py`. The resulting training, validation and testing metrics will be printed directly into the console. 

Please note that for the sake of code simplicity, some parts of the code are not runtime optimized but rather use widely adopted implementations (eg. when calculating the metrics). 