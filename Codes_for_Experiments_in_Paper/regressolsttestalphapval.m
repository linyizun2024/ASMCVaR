function [ xhat,tV,ttesttotalpval] = regressolsttestalphapval( A,y )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

%Input:
%   A: CW for the ubah strategy
%   y: CW for the PO strategy


%Output:
%   xhat(1): alpha factor
%   ttesttotalpval(1): p-value for the right-side t-test of the alpha factor


A=tick2ret(A);
y=tick2ret(y);
[n,m]=size(A);



Ac=[ones(n,1),A];

C=inv(Ac'*Ac);
xhat=C*Ac'*y;
yhat=Ac*xhat;



Q=(y-yhat)'*(y-yhat);


chi2=sqrt(diag(C)*Q/(n-m-1)); 
tV=xhat./chi2;
ttesttotalpval=tcdf(-tV,n-m-1);



end

