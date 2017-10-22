%% Assignment 2a
heart = load('heart_data.mat');

chamber = heart.chamber_values;
background = heart.background_values;

chamber_dist = fitdist(chamber, 'Normal')
background_dist = fitdist(background, 'Normal')
%% Assignment 2b
I = heart.im;

M=size(I, 1); % Height of image
N=size(I, 2); % Width of image
n = M*N; % Number of image pixels

Neighbors = edges4connected(M,N);
i=Neighbors(:,1);
j=Neighbors(:,2);


mu1 = chamber_dist.mu;
mu0 = background_dist.mu;
lambda = 0.42;
A = sparse(i,j,lambda,n,n);

T = [ (I(:)-mu1).^2 (I(:)-mu0).^2];
T = sparse(T);

[E, Theta] = maxflow(A,T);
Theta = reshape(Theta,M,N);
Theta = double(Theta);

% imshowpair(Theta, I, 'montage');
imwrite(Theta, sprintf('heart_seg_lambda%f.png', lambda));