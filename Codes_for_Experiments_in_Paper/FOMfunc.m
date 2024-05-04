function [Sharpe,MaxDD,alphafactor,Ttestpval,cvar] = FOMfunc(CW,data)

Sharpe = sharpe1self(CW);
MaxDD = maxdrawdown(CW);
[cum_ret, cumprodret, dailyret, dailyportfolio] = ubah_run_self(data);
[xhat,tV,ttesttotalpval] = regressolsttestalphapval(cumprodret,CW);
alphafactor = xhat(1);
Ttestpval = ttesttotalpval(1);
[ v,cvar ] = cvar2(CW);
end