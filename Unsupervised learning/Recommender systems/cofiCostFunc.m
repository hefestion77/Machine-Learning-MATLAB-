function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);


% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

% mean_rated = sum(Y, 2)./sum(R,2);
h = X * Theta';
J = (1/2) * (sum(sum(((h - Y).^2).*R)) + lambda * sum(sum(Theta.^2)) + lambda * sum(sum(X.^2))); % regularized collaborative filtering cost function 

X_grad = ((h - Y).*R) * Theta + lambda * X; % num_movies x num_features matrix, containing the 
                                          % partial derivatives w.r.t. to each element of X
                                          
Theta_grad = ((h - Y).*R)' * X + lambda * Theta; % num_users x num_features matrix, containing the 
                                                 % partial derivatives w.r.t. to each element of Theta

grad = [X_grad(:); Theta_grad(:)];

end
