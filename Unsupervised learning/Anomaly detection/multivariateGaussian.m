function p = multivariateGaussian(X, mu, sigma2)
% MULTIVARIATEGAUSSIAN Computes the probability density function of the multivariate gaussian distribution.

n = size(X, 2); % number of features

if size(sigma2, 1) == 1 || size(sigma2, 2) == 1
    sigma2 = diag(sigma2);
end

X = bsxfun(@minus, X, mu(:)');

p = (2 * pi) ^ (- n / 2) * det(sigma2) ^ (-0.5) * ...
            exp(-0.5 * sum(bsxfun(@times, X * pinv(sigma2), X), 2));
      
end
