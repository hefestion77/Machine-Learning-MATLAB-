% (FIRST CHECK HOW THE LEARNING ALGORITHM WORKS WITHOUT PCA...)
    
% 1) PREPROCESS DATA (ALWAYS PERFORM MEAN NORMALIZATION +/- FEATURE SCALING):
% ----------------------------------------------------------------

% X might be X_train if you have splitted the data set in X_train, X_cv and X_test

[X_norm, mu, ~] = featureNormalize(X);


% 2) COMPUTE COVARIANCE MATRIX SIGMA AND PERFORM SINGULAR VALUE DECOMPOSITION ON IT
% -------------------------------------------------------------------------
[m, n] = size(X_norm);



sigma = (1/m)*(X_norm'*X_norm); % sigma (n x n) is the covariance matrix of X
                                % Note that X_norm'*X_norm is the Gram matrix of
                                % the columns of X_norm
 
[U, S, V] = svd(sigma); % U is a unitary matrix (n x n) with column "eigenvectors"
                        % of sigma * sigma' 
                        
                        % S is a diagonal matrix (n x n) with the squared roots 
                        % of the non-zero "eigenvalues" of sigma
                        
                        % V is a unitary matrix (n x n) with the column "eigenvectors"
                        % of sigma' * sigma
                        
                        % sigma = U*S*V'
                        

                        
% Draw the eigenvectors centered at mean of data. These lines show the directions 
% of maximum variations in the dataset.
hold on;
drawLine(mu, mu + 1.5 * S(1,1) * U(:,1)', '-k', 'LineWidth', 2);
drawLine(mu, mu + 1.5 * S(2,2) * U(:,2)', '-k', 'LineWidth', 2);
hold off;

% 3) FIND THE BEST VALUE OF k (the one which keeps a 99% of the variance)... 
% choose k = 2 or 3 if you want to compress the data to plot it:

variance_retention = 0;
k = 0;

while variance_retention < 0.99
    k = k + 1;
    Ureduce = U(:,1:k); % this matrix (nxk) will contain the first K column "eigenvectors" in U
    Z = X_norm * Ureduce; % Z is a (m x k) matrix which is the projection of 
                 % the normalized inputs X into the reduced dimensional space spanned by
                 % the first K "eigenvectors" of U
                 
    variance_retention = sum(sum(S(:,1:k)))/trace(S);
end

fprintf('The number of principal components which retain a 99%% of the variance is %i \n\n', k);

% -------------------------------------------------------------------
% NOW YOU MAY USE THIS k TO GET Ureducecv, Zcv, Ureducetest and Ztest
% -------------------------------------------------------------------



% 4) THE FOLLOWING RETURNS AN APPROXIMATION OF THE NORMALIZED DATA THAT HAS BEEN REDUCED TO K DIMENSIONS
% ------------------------------------------------------------------------------------------------------
X_rec = Z * Ureduce'; % X_rec will be a m x n matrix again

%  Plot the normalized dataset (returned from pca)
plot(X_norm(:, 1), X_norm(:, 2), 'bo');
axis([-4 3 -4 3]); axis square
%  Draw lines connecting the projected points to the original points
hold on;
plot(X_rec(:, 1), X_rec(:, 2), 'ro');
for i = 1:size(X_norm, 1)
    drawLine(X_norm(i,:), X_rec(i,:), '--k', 'LineWidth', 1);
end
hold off
