K = ????  % num. of clusters (choose manually)

% 1) CHOOSE INITIAL CENTROIDS:
% -----------------------------

[m, n] = size(X);
num_iter = 50;
total_J = zeros(num_iter,1);
all_initial_centroids = [];

for j = 1:num_iter
    initial_centroids = datasample(X, K, "Replace", false);
    all_initial_centroids = [all_initial_centroids; initial_centroids];
    idx = zeros(m, 1); % cluster assigned to each training example
    min_dist = zeros(m, 1); % distance from every training example to the nearest centroid
    for i = 1:m
        dist = sum((X(i,:)-initial_centroids).^2,2); % ||x(i)-centroids||^2
        [min_dist(i),idx(i)] = min(dist, [], 1); % returns the distance from that t.example to the 
                                             % nearest centroid and the cluster assigned 
                                             % to that example
    end

    total_J(j) = mean(min_dist); % distortion cost function for the K-means algorithm
end

[~,minJ_ind] = min(total_J, [],1);
initial_centroids = all_initial_centroids(K*(minJ_ind-1)+1:(K*(minJ_ind-1))+K, :);

fprintf('The best initial centroids are (K 1xn horizontal vectors): \n\n');
fprintf('%f \t%f \n',initial_centroids');

% 2) COMPUTE K-MEANS ALGORITHM:
% ------------------------------

max_iters = 10;
centroids = initial_centroids;
previous_centroids = centroids;
idx = zeros(size(X,1), 1); % cluster assigned to each training example

for j = 1:max_iters
    
    % Output progress
    fprintf('K-Means iteration %d/%d...\n', j, max_iters);
    if exist('OCTAVE_VERSION')
        fflush(stdout);
    end
    
    for i = 1:m
    dist = sum((X(i,:)-centroids).^2,2); % ||x(i)-centroids||^2
    [~,idx(i)] = min(dist, [], 1); % returns the distance from that t.example to the 
                                             % nearest centroid and the cluster assigned 
                                             % to that example
    end

    % Optionally, plot progress here
    plotDataPoints(X, idx, K);
    hold on;

    % Plot the centroids as black x's
    plot(centroids(:,1), centroids(:,2), 'x', ...
          'MarkerEdgeColor','k', ...
          'MarkerSize', 10, 'LineWidth', 3);

    % Plot the history of the centroids with lines
    for i=1:size(centroids,1)
        drawLine(centroids(i, :), previous_centroids(i, :));
    end
    hold off;

    % Title
    title(sprintf('Iteration number %d', j))
    previous_centroids = centroids;
    fprintf('Press enter to continue.\n');
    pause;
    % end
    
    % MODIFY THE FOLLOWING DEPENDING ON THE NUMBER OF CLUSTERS K:
    Cluster1 = [];
    Cluster2 = [];
    Cluster3 = [];

    for i = 1:m
        if idx(i) == 1
            Cluster1 = [Cluster1; X(i, :)];
        elseif idx(i) == 2
            Cluster2 = [Cluster2; X(i, :)];
        else
            Cluster3 = [Cluster3; X(i, :)];
        end
    end

    mu1 = mean(Cluster1);
    mu2 = mean(Cluster2);
    mu3 = mean(Cluster3);

    centroids = [mu1; mu2; mu3];
end

% Hold off if we are plotting progress
% if plot_progress
    % hold off;
% end
