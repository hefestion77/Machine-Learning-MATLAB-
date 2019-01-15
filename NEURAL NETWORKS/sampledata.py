def sampledata(X, y = None):
    """
    Shuffles 10 times ndarrays X and y and returns
    3 different datasets; training examples, cross-validation examples and
    test examples with their correspondent y values (Xtrain, ytrain, Xval, 
    yval, Xtest, ytest)
    
    If y is not passed; the function will only return 60 % of training examples, 
    20 % of cross-validation examples and 20 % of test examples from X.
    
    """
    import numpy as np
    
    if y is None:
        m = X.shape[0]
    
        for i in range(10):
            np.random.shuffle(X)

        num_train_examp = int(np.floor(m * 0.6))
        num_cv_examp = int(np.floor((m - num_train_examp) / 2))
                     # num_test_examp = m - num_train_examp - num_cv_examp;

        train_dataset = X[:num_train_examp]
        cv_dataset = X[num_train_examp:num_train_examp + num_cv_examp]
        test_dataset = X[num_train_examp + num_cv_examp:]

        return train_dataset, cv_dataset, test_dataset
        
    else:
        m = X.shape[0]
        dataset = np.hstack((X,y))
        
        for i in range(10):
            np.random.shuffle(dataset)
    
        num_train_examp = int(np.floor(m * 0.6))
        num_cv_examp = int(np.floor((m - num_train_examp) / 2))
                         # num_test_examp = m - num_train_examp - num_cv_examp;
    
        train_dataset = dataset[:num_train_examp]
        cv_dataset = dataset[num_train_examp:num_train_examp + num_cv_examp]
        test_dataset = dataset[num_train_examp + num_cv_examp:]
    
        Xtrain = train_dataset[:,:-1]
        ytrain = train_dataset[:,-1:]
    
        Xval = cv_dataset[:,:-1]
        yval = cv_dataset[:,-1:]
    
        Xtest = test_dataset[:,:-1]
        ytest = test_dataset[:,-1:]
    
        return Xtrain, ytrain, Xval, yval, Xtest, ytest
