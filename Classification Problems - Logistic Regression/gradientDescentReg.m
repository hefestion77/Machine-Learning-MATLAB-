function [theta, J_history] = gradientDescentReg(X,y,theta,lambda,alpha,num_iters)
% By applying the regularization parameter lambda (p.e. when 
% features (n) >> training examples (m)),
% GRADIENTDESCENTREG Performs gradient descent to learn theta for logistic regression
% theta = GRADIENTDESCENTREG(x, y, theta, alpha, num_iters) updates theta by 
% taking num_iters gradient steps with learning rate alpha

m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
   h = (1 + (exp(-(X*theta)))).^(-1); % hypothesis function
   thetaToReg = theta;
   thetaToReg(1) = [];
   theta = theta - (alpha * (((X'* (h - y)) / m) + [0; ((lambda/m)*thetasToReg)]);
   J_history(iter) = costFunctionReg(X,y,theta,lambda);
end

end
