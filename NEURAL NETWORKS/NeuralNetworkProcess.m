% PROCESS TO IMPLEMENT A NEURAL NETWORK WITH 1 HIDDEN LAYER
% ----------------------------------------------------------

input_layer_size = ???; % num of features % 400
hidden_layer_size = ??; % 25
num_labels = ??; % num of classes (labels) % 10
m = size(X, 1); % num of training examples

% 1) INITIALIZE RANDOM THETAS FOR THE INITIAL FORWARD PROPAGATION

Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

%  Unroll parameters
initial_nn_params = [Theta1(:) ; Theta2(:)];

% FORWARD PROPAGATION TO COMPUTE VALUES FOR THE OUTPUT LAYER (h):

a1 = [ones(m, 1) X];
z2 = a1*Theta1';
a2 = [ones(size(z2, 1), 1) sigmoid(z2)];
z3 = a2*Theta2';
h = sigmoid(z3);

% transform the values of vector y into a matrix with "num_labels" columns with 
% values 1 or 0 and m rows;

classes = 1:num_labels;
unrolled_y = (y == classes);

% REGULARIZED COST FUNCTION;

Theta1toReg = Theta1(:, 2:end);
Theta2toReg = Theta2(:, 2:end);

lambda = 1;
J = (1/m)*(sum(sum((-unrolled_y).*log(h) - (1-unrolled_y).*log(1-h)))) + (lambda/(2*m)) * (sum(sum(Theta1toReg.^2))+sum(sum(Theta2toReg.^2)));

% COMPUTE deltas and accumulative matrices DELTAS
delta3 = h - unrolled_y;
delta2 = (delta3*Theta2).* sigmoidGradient([ones(size(z2, 1), 1) z2]);
delta2 = delta2(:, 2:end);


DELTA2 = delta3'*a2; 
DELTA1 = delta2'*a1;

% GRADIENTS OF THETA1 AND THETA2
Theta1_grad = DELTA1./m + (lambda/m)*[zeros(size(Theta1,1), 1) Theta1toReg];
Theta2_grad = DELTA2./m + (lambda/m)*[zeros(size(Theta2,1), 1) Theta2toReg];


% UNROLL GRADIENTS
grad = [Theta1_grad(:) ; Theta2_grad(:)];

% NOW PERFORM GRADIENT CHECKING WITH lambdatest = 3 FOR EXAMPLE;
checkNNGradients(lambdatest);

% ----------------------------
% THE FOLLOWING IS TO BE USED ONLY IF WE WANT TO USE FMINCG FUNCTION

options = optimset('GradObj', 'on', 'MaxIter', 500);

%Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, ~] = fmincg(costFunction, initial_nn_params, options);

% ----------------------------------

% OBTAIN THETA1 AND THETA2 BACK FROM nn_params AND ACCURACY

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), num_labels, (hidden_layer_size + 1));

pred = predict(Theta1, Theta2, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);
end
