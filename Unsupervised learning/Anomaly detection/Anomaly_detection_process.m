% ANOMALY DETECTION PROCESS (MODIFY IF NECESSARY)
% ------------------------------------------

% - FAST CHECK TO SEE WHETHER DATA HAS A NORMAL (GAUSSIAN DISTRIBUTION):

hist(X, 50);

% 1) PLOT DATA (MODIFY IF NECESSARY):

plot(X(:, 1), X(:, 2), 'bx');
axis([0 30 0 30]);
xlabel('????');
ylabel('?????');
title('???');

% 2) ESTIMATE THE PARAMETERS FOR THE DENSITY FUNCTION P OF THE UNIVARIATE GAUSSIAN DISTRIBUTION (if num of train. examples <<<<<<< features):

[mu, sigma2] = estimateGaussian(X); % mu (means) and sigma2 (variances) will be 1 x n vectors
p = GaussianDensity(X, mu, sigma2); 

% OR ESTIMATE THE PARAMETERS FOR THE DENSITY FUNCTION P 
% OF A MULTIVARIATE GAUSSIAN DISTRIBUTION (if num of train. examples >>>>>>>> features):

[mu, sigma2] = estimateMultiGaussian(X); % mu (means) will be a 1 x n vector, sigma2 the covariance matrix n x n of X 
p = multivariateGaussian(X, mu, sigma2);


% 3) FIND, USING THE CROSS-VALIDATION SET, THE BEST THRESHOLD (EPSILON) WITH THE HIGHEST F1 SCORE:

pval = multivariateGaussian(Xval, mu, sigma2);

[epsilon, F1] = selectThreshold(yval, pval);
fprintf('Best epsilon found using cross-validation: %e\n', epsilon);
fprintf('Best F1 on Cross Validation Set:  %f\n', F1);


% 4) FIND THE OUTLIERS IN THE TRAINING SET:

outliers = find(p < epsilon); % finds the linear indices of the values of the vector p that are
                              % lower than epsilon
                              
fprintf('# Outliers found: %d\n', sum(p < epsilon));

% 5) VISUALIZE THE FIT:

visualizeFit(X, mu, sigma2);
xlabel('????');
ylabel('????');
hold on

% 6) DRAW A RED CIRCLE AROUND THOSE OUTLIERS

plot(X(outliers, 1), X(outliers, 2), 'ro', 'LineWidth', 2, 'MarkerSize', 10);
hold off
