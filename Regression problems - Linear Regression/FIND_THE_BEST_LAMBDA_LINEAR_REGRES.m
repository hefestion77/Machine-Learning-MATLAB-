% THESE ARE THE STEPS TO FIND THE BEST LAMBDA TO REGULARIZE A
% LINEAR REGRESSION PROBLEM
% -------------------------------------------------------------------


% REMEMBER TO ADD A COLUMN OF ONES FOR X0 TO INTERCEPT THETA0
% IF NOT DONE BEFORE

m = size(X,1); % num of training examples

% FEATURE NORMALIZE:

[X,~,~] = featureNormalize(X);


% Selection of possible values of lambda from 0 to 10.24
lambda_vec = [0:0.2:10.24]';


iterations = 50;
error_train = zeros(length(lambda_vec), 1);
error_val = zeros(length(lambda_vec), 1);

for i = 1:length(lambda_vec)
    error_train_iter = zeros(iterations,1);
    error_val_iter = zeros(iterations,1);
    for j = 1:iterations
        [Xtrain, ytrain, Xval, yval, Xtest, ytest] = sampledata(X, y);
        mtrain = size(Xtrain,1);
        mval = size(Xval,1);
        
        thetatrain = trainLinearReg(Xtrain, ytrain, lambda_vec(i));
        htrain = Xtrain*thetatrain;
        errortrain = (1/(2*mtrain))*(((htrain-ytrain)'*(htrain-ytrain)));
        error_train_iter(j) = errortrain;
    
        hval = Xval*thetatrain;
        errorval = (1/(2*mval))*(((hval-yval)'*(hval-yval)));
        error_val_iter(j) = errorval;
    end
    error_train(i) = mean(error_train_iter);
    error_val(i) = mean(error_val_iter);
end

% TO CHOOSE THE BEST LAMBDA, FIND THE LOWEST ERROR_VAL FOR EVERY VALUE OF LAMBDA HERE:

plot(lambda_vec, error_train, lambda_vec, error_val);
legend('Train', 'Cross Validation');
xlabel('lambda');
ylabel('Error');

for i = 1:length(lambda_vec)
    if i == 1
        fprintf('lambda\t\tTrain Error\tValidation Error\n');
    end
    fprintf('%f\t%f\t%f\n',lambda_vec(i), error_train(i), error_val(i));
end

[~, ind] = min(error_val);
best_lambda = lambda_vec(ind);

fprintf('\nThe best Lagrange multiplier lambda is %f', best_lambda);

% NOW COMPUTE ERROR_TEST WITH THE BEST LAMBDA YOU FOUND:

% lambda = ?;
% mtest = size(Xtest,1);
% besttheta = trainLinearReg(X, y, lambda);

% htest = Xtest*besttheta;
% error_test = (1/(2*mtest))*(((htest-ytest)'*(htest-ytest)));
% fprintf('Test error = %f', error_test);

end
