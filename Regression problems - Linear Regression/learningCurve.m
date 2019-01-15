function [error_train, error_val] = ...
    learningCurve(X, y, Xval, yval, lambda)
% This function returns the train and
% cross validation set errors for a learning curve. In particular, 
% it returns two vectors of the same length - error_train and 
% error_val. Then, error_train(i) contains the training error for
% i examples (and similarly for error_val(i)).
%
%   In this function, you will compute the train and test errors for
%   dataset sizes from 1 up to m. In practice, when working with larger
%   datasets, you might want to do this in larger intervals.
%

% Xsample = randsample(X, 8)

m = size(X, 1); % Number of training examples
mval = size(Xval,1); % Number of validation examples

error_train = zeros(m, 1);
error_val   = zeros(m, 1);

for i = 1:m
    Xtrain = X(1:i,:);
    ytrain = y(1:i);
    thetatrain = trainLinearReg(Xtrain, ytrain, lambda);
    
    htrain = Xtrain*thetatrain;
    errortrain = (1/(2*i))*(((htrain-ytrain)'*(htrain-ytrain)));
    error_train(i) = errortrain;
    
    hval = Xval*thetatrain;
    errorval = (1/(2*mval))*(((hval-yval)'*(hval-yval)));
    error_val(i) = errorval;
end

end
