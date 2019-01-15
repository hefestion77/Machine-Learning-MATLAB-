function [J, grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));


m = size(X, 1); % num of training examples

% FORWARD PROPAGATION TO COMPUTE VALUES FOR THE OUTPUT LAYER (h):

a1 = [ones(m, 1) X];
z2 = a1*Theta1';
a2 = [ones(size(z2, 1), 1) sigmoid(z2)];
z3 = a2*Theta2';
h = sigmoid(z3);

% transform y into a matrix with "num_labels" columns with values 1 or 0;

classes = 1:num_labels;
unrolled_y = (y == classes);

% REGULARIZED COST FUNCTION;

Theta1toReg = Theta1(:, 2:end);
Theta2toReg = Theta2(:, 2:end);


J = (1/m)*(sum(sum((-unrolled_y).*log(h) - (1-unrolled_y).*log(1-h)))) + (lambda/(2*m)) * (sum(sum(Theta1toReg.^2))+sum(sum(Theta2toReg.^2)));

% COMPUTE deltas and accumulative matrices DELTAS
delta3 = h - unrolled_y;
delta2 = (delta3*Theta2).* sigmoidGradient([ones(size(z2, 1), 1) z2]);
delta2 = delta2(:, 2:end); % to skip all delta2sub0s


DELTA2 = delta3'*a2;
DELTA1 = delta2'*a1;

% REGULARIZED GRADIENTS OF THETA1 AND THETA2
Theta1_grad = DELTA1./m + (lambda/m)*[zeros(size(Theta1,1), 1) Theta1toReg];
Theta2_grad = DELTA2./m + (lambda/m)*[zeros(size(Theta2,1), 1) Theta2toReg];


% UNROLL GRADIENTS
grad = [Theta1_grad(:) ; Theta2_grad(:)];

% ----------------------------
% THE FOLLOWING IS TO BE USED IF WE WANT TO USE FMINCG FUNCTION

% options = optimset('MaxIter', 50);
% lambda = 1;

%Create "short hand" for the cost function to be minimized
%costFunction = @(p) nnCostFunction(p, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
%[nn_params, ~] = fmincg(costFunction, initial_nn_params, options);

% ----------------------------------

% OBTAIN THETA1 AND THETA2 BACK FROM nn_params AND ACCURACY

% Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));
% Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), num_labels, (hidden_layer_size + 1));

% m = size(X, 1); % num of training examples
% num_labels = size(Theta2, 1); % num of classes (labels)

% h1 = sigmoid([ones(m, 1) X] * Theta1');
% h2 = sigmoid([ones(m, 1) h1] * Theta2');
% pred = max(h2, [], 2);

% fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);
end