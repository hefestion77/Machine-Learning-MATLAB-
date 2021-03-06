function p = GaussianDensity(X, mu, sigma2)
% GaussianDensity Computes the probability density function of the
% Gaussian distribution in a m x 1 vector (m is the number of examples)

%   The input X is the dataset with each n-dimensional data point in one row
%   The output is an n-dimensional vector mu, the mean of the data set
%   and the variances in the diagonal matrix sigma2
% 

p = prod((sqrt(2*pi*sigma2).^(-1)).* exp(-((X-mu).^2)./(2*sigma2)), 2); % probability density function of the
                                                                       % Gaussian distribution
end
