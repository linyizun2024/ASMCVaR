function [opt_loss,opt_supp,opt_beta] = L0regress(X,y,m)
%% Input variables
% X: Feature matrix of training data;
% y: Label of training data.
% Model: 1/2*norm(X*beta-y,2)^2+iota_m(beta)

%% Output variables
% beta: Linear regression coefficients

%% Brute-force method for L0 regression
opt_loss = inf;
n = size(X,2);
opt_supp = zeros(1,n);
opt_beta = zeros(n,1);
allcomb = combnk(1:n,m);
for i = 1:size(allcomb,1)
    support = allcomb(i,:);
    suppX = X(:,support);
    suppbeta = suppX'*suppX\(suppX'*y);
    beta = zeros(n,1);
    beta(support) = suppbeta;
    lossvalue = 1/2*norm(X*beta-y,2)^2;
    if lossvalue<opt_loss
        opt_loss = lossvalue;
        opt_supp = support;
        opt_beta = beta;
    end
end

end

