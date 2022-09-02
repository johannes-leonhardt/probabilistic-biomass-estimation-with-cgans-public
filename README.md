# Probabilistic Biomass Estimation with Conditional Generative Adversarial Networks

Johannes Leonhardt, Lukas Drees, Peter Jung, Ribana Roscher

This repository contains code and data samples regarding our paper "Probabilistic Biomass Estimation with Conditional Generative Adversarial Networks", published at GCPR 2022.

## Abstract

Biomass is an important variable for our understanding of the terrestrial carbon cycle, facilitating the need for satellite-based global and continuous monitoring. However, current machine learning methods used to map biomass can often not model the complex relationship between biomass and satellite observations or cannot account for the estimation's uncertainty. In this work, we exploit the stochastic properties of Conditional Generative Adversarial Networks for quantifying aleatoric uncertainty. Furthermore, we use generator Snapshot Ensembles in the context of epistemic uncertainty and show that unlabeled data can easily be incorporated into the training process. The methodology is tested on a newly presented dataset for satellite-based estimation of biomass from multispectral and radar imagery, using lidar-derived maps as reference data. The experiments show that the final network ensemble captures the dataset's probabilistic characteristics, delivering accurate estimates and well-calibrated uncertainties.
