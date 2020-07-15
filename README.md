# Bring the GANs into Action - Extending DeepOBS with novel test problems

## 📇 Table of Contents


- [Bring the GANs into Action - Extending DeepOBS novel test problems.](#https://github.com/Vanessa-Ts/DeepOBS_BA)
  - [ Introduction][# Introduction]
  - [📦 Extend Data DeepOBS][# Extend]
  - [ Results][# Results]
  - [📦 Runners][ # Runners]
  - [📦 Groundwork for the FID calculation][ # Groundwork ]

## Introduction
[# Introduction]: #Introduction
In this repository a selection of the implementations and results, obtained during my thesis are provided.

This work is a direct extension of the development branch of **DeepOBS** - A deep learning optimizer benchmark suite.
Including the additional data sets **Animal Faces HQ** (Adapted from https://github.com/clovaai/stargan-v2) and **CelebA** (source: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).
Further the results from the experiments are provided, to illustrate the performance of the novel test problems.

For the reproduction of this work, scripts to directly run the optimizers on the test problems are provided.

Lastly, the repository contains the preparations, for a future implementation of the **Fréchet Inception Distance**.

-----------------------------------------------------------------------------------------------------------------------
## Extending Data DeepOBS
[# Extend]: #Extend
Most of the data sets provided in **DeepOBS**, the images are downloaded and preprocessed automatically.
For the the data sets, that have been newly introduced, there is no reliable source for the automatized procedure, 
therefore the data sets are offered in folders, that can be directly used with the instances of the test problems.

-----------------------------------------------------------------------------------------------------------------------

## Results
[# Results]: #Results
Within this thesis, various experiments have been done, with the novel test problems, 
in order to investigate their performance and weaknesses, in the training process, with different optimizers.
The results of each test problem are saved in folders respectively, 
that follow the original structure of the **DeepOBS** code.


-----------------------------------------------------------------------------------------------------------------------

## Runners
[ # Runners]: #Runners
To simplify the execution of the test problems, 
this folder contains scripts for **SGD**, **MOMENTUM** and **ADAM** respectively.
The required arguments can be directly inserted. Additionally, it provides an easy way to switch 
between manually setting the batch size and epochs for the run or using the default setting.

-----------------------------------------------------------------------------------------------------------------------

## Groundwork for the FID calculation
[ # Groundwork ]: #Groundwork
Due to the limited time of a bachelors thesis,
a full implementation of the Fréchet Inception Distance, cannot be provided,
in the first introduction of **GAN**s in **DeepOBS**.
However offering a quantitative evaluation method, in the automatized benchmark procedure of **DeepOBS**, is crucial for optimizer comparison.
Therefore several preparations have been done, to smoothen the implementation of such.
These include, the pre-trained inception-v3 network and the basic calculations needed for the **FID**,
as well as some adjustments in the training process of the **GAN**s in **DeepOBS**.






