clear;
close all;
clc;
%% Setting of parameters
ny = 50;
n = 10;
m = 3;
gamma = 1E-7;
initbeta = zeros(n,1);
ITER = 5000000;

%% Generate the data for regression
[X,y] = LRdatagen(ny,n,m);

%% Call regression functions
[opt_loss_L0,opt_supp_L0,opt_beta_L0] = L0regress(X,y,m);
[opt_loss_RelaxL0,opt_supp_RelaxL0,opt_beta_RelaxL0,opt_eta_RelaxL0] = RelaxL0regress(X,y,m,gamma);
[all_loss_PALM,beta_PALM,eta_PALM] = PALMforRelaxL0regress(X,y,m,gamma,initbeta,ITER);
final_loss_PALM = all_loss_PALM(ITER);
final_supp_PALM = find(beta_PALM>1E-6);

figure;
plot(1:ITER,all_loss_PALM);
