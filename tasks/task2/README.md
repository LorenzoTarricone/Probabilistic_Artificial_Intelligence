# Bayesian Neural Networks for Satellite Image Classification

The project involves implementing a Bayesian Neural Network (BNN) for well-calibrated predictions in land-use classification using satellite images. 
The BNN is trained using Stochastic Weight Averaging (SWA) and SWA-Gaussian (SWAG) for approximate inference, with a focus on uncertainty estimation in critical 
domains. The dataset consists of 60×60 GB satellite images, with the training set containing well-defined images and the test set featuring ambiguous ones. 
The goal is not only land-use classification but also identifying ambiguous images based on predicted confidence. The specific tasks addressed involve the 
implementation of SWAG-Diagonal and SWAG-Full method, and exploring different approaches to improve model calibration and prediction costs. 
More specifically, the SWAG approach involves tracking running first and second moments, and for the full SWAG, maintaining a lower-rank approximation of 
the standard deviation matrix. These estimates are then utilized to gather a posterior sample by re-scaling and re-centering samples from a standard 
Gaussian distribution.
