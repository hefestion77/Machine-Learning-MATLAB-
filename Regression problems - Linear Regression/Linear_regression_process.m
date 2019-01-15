% PROCESS TO COMPUTE LINEAR REGRESSION:
% --------------------------------------

% 1) FEATURE NORMALISE AND SCALING:

[X, mu, sigma] = featureNormalize(X); % wil return normalised and scaled values of the features in X
                                      % according to their respective means and standard deviations.
                                      
% 2) ADD 1s TO X TO INTERCEPT THETA0

X = [ones(m, 1) X];

% ------------------------------------------------------------------
% CHECK FIND_THE_BEST_LAMBDA_LINEAR_REGRES.M TO FIND THE BEST LAMBDA
% -------------------------------------------------------------------

% 3) COMPUTE COST AND GRADIENT FOR LINEAR REGRESSION:

theta = zeros(size(X, 2), 1); % initial values of the weights
lambda = ?;
[J, grad] = linearRegCostFunction(X, y, theta, lambda);

%  4) Train linear regression with function fmincg:

[theta] = trainLinearReg(X, y, lambda);

%  5) Plot fit over the data

figure;
plot(X, y, 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
xlabel('(x)');
ylabel('(y)');
hold on;
plot(X, X*theta, '--', 'LineWidth', 2)
hold off;


% 6) Learning curve

[Xtrain, ytrain, Xval, yval, Xtest, ytest] = sampledata(X, y);
[error_train, error_val] = learningCurve(X, y, Xval, yval, lambda);

plot(1:m, error_train, 1:m, error_val);
title('Learning curve for linear regression')
legend('Train', 'Cross Validation')
xlabel('Number of training examples')
ylabel('Error')
axis([0 13 0 150])


%--------------------------------------------------------------------------------------------
% -------------YOU MAY USE GRADIENT DESCENT TO FIND THE BEST THETAS TO MIN THE COST FUNCION INSTEAD

% ----------CHOOSE LEARNING RATE ALPHA (the best one will minimize J sooner, see the graph
% for every possible alpha)

alpha_vec = [0.01, 0.03, 0.1, 0.3, 1, 3];
num_iters = 200; % this may be modified later depending on the graph

theta = zeros(size(X,2), 1);
[~, J_history] = gradientDescentMulti(X, y, theta, alpha, lambda, num_iters);

% Plot the convergence graph
plot(1:num_iters, J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');


alpha = ???;
num_iters = ????;
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, lambda, num_iters);
% fprintf('Theta computed from gradient descent:\n%f,\n%f',theta(1),theta(2)) % modify if necessary according to number of thetas

% OPTIONAL: PLOT THE CONVERGENCE GRAPH (to see how cost J decreases with every iteration until convergence)

plot(1:num_iters, J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');
