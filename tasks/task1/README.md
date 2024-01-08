# Gaussian Process Regression for Air Pollution Prediction

To complete Task 1, we applied Gaussian Processes for predicting PM2.5 levels. First of all, data exploration revealed a non-uniform data distribution on 
the \[0,1\]^2 domain. To handle the large dataset, we subsampled it using both a uniform and a k-means clustering approach. The latter one appeared to be better 
capture the distribution of the data and efficiently reduce the computational cost. We then used a validation set to select the kernel and fine-tune its 
hyperparameters, ultimately achieving the best performance with a Matern-3/2 kernel. Finally, we address the challenge of an asymmetric cost function. We knew that 
minimizing the loss resulted in an analytical solution, which is the mean of the posterior distribution, however, the cost function's asymmetry for the candidate 
residential areas led us to introduce a penalty for those areas, so to mitigate the risk of underpredictions. This penalty is a fraction (treated as an 
additional hyperparameter) of the posterior standard deviation. This comprehensive approach yielded a robust GP regression model for predicting PM2.5 
concentrations.
