function [all_loss,beta,eta] = PALMforRelaxL0regress(X,y,m,gamma,initbeta,ITER)
%% Input variables
% X: Feature matrix of training data;
% y: Label of training data.
% Model: 1/2*norm(X*beta-y,2)^2+1/(2*gamma)*norm(beta-eta,2)^2+iota_m(eta)

%% Output variables
% beta: Linear regression coefficients

%% PALM for Envelope L0 regression
n = size(X,2);
beta = initbeta;
eta = beta;
alpha1 = 0.99/norm(X'*X+eye(n)/gamma,2);
alpha2 = 0.99*gamma;
all_loss = zeros(ITER,1);

for k = 1:ITER
    gradH_beta = X'*(X*beta-y)+(beta-eta)/gamma;
    beta = beta-alpha1*gradH_beta;
    eta = (1-alpha2/gamma)*eta+alpha2/gamma*beta;
    [~, sorted_ind] = sort(abs(eta),'descend');
    eta(sorted_ind(m+1:n)) = 0;
    all_loss(k) = 1/2*norm(X*beta-y,2)^2;
end

end

