function [ idx, idxneg,alweight ] = testweight( day_weight_total )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
test=sum(day_weight_total);
idx=find(abs(test-1)>1e-3);
alweight=test(idx);
idxneg=find(day_weight_total<0);
end

