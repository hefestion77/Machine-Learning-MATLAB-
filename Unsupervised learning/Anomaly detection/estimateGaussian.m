function [mu, sigma2] = estimateGaussian(X)
% ESTIMATEGAUSSIAN This function estimates the parameters of a 
% Gaussian distribution using the data in X.

%   The input X is the dataset with each n-dimensional data point in one row
%   The output is an n-dimensional vector mu, the mean of the data set
%   and the variances in the diagonal matrix sigma2
% 

mu = mean(X); % 1 x n vector
sigma2 = var(X, 1); % variance calculated from 1/m... 1 x n vector

end
