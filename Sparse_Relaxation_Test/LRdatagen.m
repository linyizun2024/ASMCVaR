function [X,y] = LRdatagen(ny,n,m)
%% Input variables
% ny: number of samples (size of y)
% n: number of fetures (number of columns of X)
% m: sparsity of beta
% Model: y = X*beta+epsilon

%% Output variables
% beta: Linear regression coefficients

%% Setting of parameters
rho = 0.5;
sigma = 1;
rep = 1;

%% Simulation setup
Sigma_X = eye(n);
for i = 2:n
    for j = 1:(i-1)
        Sigma_X(i,j) = rho^abs(i-j);
        Sigma_X(j,i) = Sigma_X(i,j);
    end
end
    
%% Set up coef vector
beta_sig = ones(rep*m, 1);
n_sig = length(beta_sig);
beta = [beta_sig; zeros(n-n_sig,1)];
    
%% Training data
X = mvnrnd(zeros(n,1),Sigma_X,ny); % n*p
y = X*beta+sigma*randn(ny,1); % n*1

%% Test data
% XT = mvnrnd(zeros(n,1), Sigma_X, ny); % n*p
% yt = XT * beta + sigma * randn(ny,1); % n*1
% ytt = XT * beta ; % no-noise version

