function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, lambda, num_iters)
% GRADIENTDESCENTMULTI Performs gradient descent to learn theta

m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    h = X*theta;
    ThetatoReg = theta(2:end);
    grad = (1/m)*((X'*(h-y)) + lambda*[0;ThetatoReg]);
    theta = theta - (alpha * grad);
    J_history(iter) = (1/(2*m))*(((h-y)'*(h-y)) + lambda*(ThetatoReg'*ThetatoReg));
end
end
