clear;
close all;
clc;

%% Import Dataset

DataSet1 = 'FF25EUnew';
DataSet2 = 'FF25new';
DataSet3 = 'FF32new';
DataSet4 = 'FF49new';
DataSet5 = 'FF100MEINVnew';
DataSet6 = 'FF100MEOPnew';
DataSet7 = 'FF100new';

DatasetName = DataSet1;
addr = ['.\DataSets\' DatasetName '.mat'];

load(addr);



fullR = (data-1);
[T,d] = size(fullR);

%% Preset parameters
Param.winsize = 60;
Param.trancost = 0/100;

Param.kappa = 1;
Param.MaxIter1 = 10000;  % 
Param.MaxIter2 = 200;  % 

Param.tol_2 = 0.0001 ; % 
Param.tol_1 = 0.001 ; % 
Param.m = 10; % 
Param.c = 0.99; % 
Param.rho = 0.02;


%---------------------------------------------------------------------------------------------------------------------------------------------------
%% Run Core Program and obtain Results
[Paramout, msMCVaR_CW, MCVaR_allw,t,runout] = PALMstrategy(Param,data);
warning off
[msMCVaR_Sharpe,msMCVaR_MaxDD,msMCVaR_alphafact,msMCVaR_Ttestpval] = FOMfunc(msMCVaR_CW,data);
warning on

