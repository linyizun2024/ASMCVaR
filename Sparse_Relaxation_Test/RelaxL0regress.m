function [opt_loss,opt_supp,opt_beta,opt_eta] = RelaxL0regress(X,y,m,gamma)
%% Input variables
% X: Feature matrix of training data;
% y: Label of training data.
% Model: 1/2*norm(X*beta-y,2)^2+1/(2*gamma)*norm(beta-eta,2)^2+iota_m(eta)

%% Output variables
% beta: Linear regression coefficients

%% Brute-force method for Envelope L0 regression
opt_loss = inf;
n = size(X,2);
ny = numel(y);
beta = zeros(n,1);
eta = zeros(n,1);
opt_supp = zeros(1,n);
opt_beta = zeros(n,1);
allcomb = combnk(1:n,m);
tItranstI = eye(n+m);
tItranstI(1:m,n+1:n+m) = -eye(m);
tItranstI(n+1:n+m,1:m) = -eye(m);

for i = 1:size(allcomb,1)
    supp = allcomb(i,:);
    nonsupp = 1:n;
    nonsupp(supp) = 0;
    nonsupp(nonsupp==0) = [];
    tX = zeros(ny,n+m);
    tX(:,1:m) = X(:,supp);
    tX(:,m+1:n) = X(:,nonsupp);
    v = (tX'*tX+tItranstI/gamma)\(tX'*y);
    beta(supp) = v(1:m);
    beta(nonsupp) = v(m+1:n);
    eta(supp) = v(n+1:n+m);
    lossvalue = 1/2*norm(X*beta-y,2)^2;
 %   objvalue = 1/2*norm(tX*v-y,2)^2+1/(2*gamma)*(norm(nonsuppbeta,2)^2+norm(suppbeta-suppeta,2)^2);
    if lossvalue<opt_loss
        opt_loss = lossvalue;
        opt_supp = supp;
        opt_beta = beta;
        opt_eta = eta;
    end
end

end

