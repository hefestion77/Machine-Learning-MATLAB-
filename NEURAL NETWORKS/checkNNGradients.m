function checkNNGradients(lambdatest)
%CHECKNNGRADIENTS Creates a small neural network to check the
%backpropagation gradients
%   CHECKNNGRADIENTS(lambdatest) Creates a small neural network to check the
%   backpropagation gradients, it will output the analytical gradients
%   produced by your backprop code and the numerical gradients (computed
%   using computeNumericalGradient). These two gradient computations should
%   result in very similar values.
%

if ~exist('lambda', 'var') || isempty(lambdatest)
    lambdatest = 0;
end

input_layer_sizetest = 3;
hidden_layer_sizetest = 5;
num_labelstest = 3;
mtest = 5; % num of training examples of the test

% We generate some 'random' test data
Theta1test = debugInitializeWeights(hidden_layer_sizetest, input_layer_sizetest);
Theta2test = debugInitializeWeights(num_labelstest, hidden_layer_sizetest);
% Reusing debugInitializeWeights to generate test X
Xtest  = debugInitializeWeights(mtest, input_layer_sizetest - 1);
ytest  = 1 + mod(1:mtest, num_labelstest)';

% Unroll test parameters
nn_paramstest = [Theta1test(:) ; Theta2test(:)];

% Short hand for cost function
costFunc = @(p) nnCostFunction(p, input_layer_sizetest, hidden_layer_sizetest, ...
                               num_labelstest, Xtest, ytest, lambdatest);

[cost, grad] = costFunc(nn_paramstest);
numgrad = computeNumericalGradient(costFunc, nn_paramstest);

% Visually examine the two gradient computations.  The two columns
% you get should be very similar. 
disp([numgrad grad]);
fprintf(['The above two columns you get should be very similar.\n' ...
         '(Left-Your Numerical Gradient, Right-Analytical Gradient)\n\n']);

% Evaluate the norm of the difference between two solutions.  
% If you have a correct implementation, and assuming you used EPSILON = 0.0001 
% in computeNumericalGradient.m, then diff below should be less than 1e-9
diff = norm(numgrad-grad)/norm(numgrad+grad);

fprintf(['If your backpropagation implementation is correct, then \n' ...
         'the relative difference will be small (less than 1e-9). \n' ...
         '\nRelative Difference: %g\n'], diff);

end

