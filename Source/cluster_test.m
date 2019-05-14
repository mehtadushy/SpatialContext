close all;
rng('default'); % For reproducibility
dim = 3
X = [randn(100,dim)*0.75+ones(100,dim);
    randn(100,dim)*0.55-ones(100,dim)];
figure;
plot(X(:,1),X(:,2),'.');
title('Randomly Generated Data');

% perform clustering
options = zeros(1,14);
options(1) = 1; % display
options(2) = 1;
options(3) = 0.1; % precision
options(5) = 1; % initialization
options(14) = 100; % maximum iterations

clusters = 2
centers = zeros(clusters, size(X,2));
[dic, ~, post] = kmedoids_custom(centers, X, options , @intersectionDistance)

figure;
plot(X(post(:,1)==1,1), X(post(:,1)==1,2), 'ro')
hold on
plot(X(post(:,2)==1,1), X(post(:,2)==1,2), 'go')
