function [Xtrain, ytrain, Xval, yval, Xtest, ytest] = sampledata(X, y)
% SAMPLEDATA(X,y) shuffles the whole dataset 10 times and splits it 
% into 3 different datasets; training examples, cross-validation examples and test
% with their correspondent y values.

m = size(X,1);
dataset = [X y];

for i = 1:10
    dataset = datasample(dataset, m, 'Replace', false);
end

num_train_examp = floor(m * 0.6);

num_cv_examp = floor((m - num_train_examp) / 2);
% num_test_examp = m - num_train_examp - num_cv_examp;

train_dataset = dataset(1:num_train_examp, :);
cv_dataset = dataset(num_train_examp + 1:num_train_examp + num_cv_examp, :);
test_dataset = dataset(num_train_examp + num_cv_examp + 1:end, :);

Xtrain = train_dataset(:, 1:end-1);
ytrain = train_dataset(:, size(train_dataset, 2));

Xval = cv_dataset(:, 1:end-1);
yval = cv_dataset(:, size(cv_dataset, 2));

Xtest = test_dataset(:, 1:end-1);
ytest = test_dataset(:, size(test_dataset, 2));


end
