function weights = xavier_he_init(L_in, L_out)
    
    % Returns an array (L_out x (L_in + 1)) of random initial weights using
    % Xavier/He initialisation method, where L_in
    % is the number of input units and L_out the number of output units 
    % in the layers adjacent to the matrix of weights
    
    % run'pkg load statistics' from the Octave prompt,if not loaded before, to
    % use the function normrnd
    
    xav_std = sqrt(sqrt(2/(L_in + L_out)));
    weights = normrnd(0, xav_std, [L_out, 1 + L_in]); 