function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
% This function computes cost and gradient for regularized linear 
% regression with multiple variables.

m = length(y); % number of training examples

ThetatoReg = theta(2:end);

h = X*theta;
J = (1/(2*m))*(((h-y)'*(h-y)) + lambda*(ThetatoReg'*ThetatoReg));

grad = (1/m)*((X'*(h-y)) + lambda*[0;ThetatoReg]);

grad = grad(:);

end
