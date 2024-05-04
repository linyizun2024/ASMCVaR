function [cum_ret, cumprod_ret, daily_ret, daily_portfolio] = ubah_run_self( data)

tc=0;
[n, m]=size(data);

% Variables for return, start with uniform weight
cum_ret = 1;
cumprod_ret = ones(n, 1);
daily_ret = ones(n, 1);

% portfolio at the beinning (end) of a period
day_weight = ones(m, 1)/m;  
day_weight_o = zeros(m, 1);
daily_portfolio = zeros(n, m);

% print log file head


% Backtests
for t = 1:1:n
    
    % Calculate t's portfolio at the beginning of t-th trading day
    if (t >= 6)
        [day_weight] = ubah_kernel(data(1:t-1, :), day_weight_o);
    end
    
    % Normalize the constraint, always useless
    day_weight = day_weight./sum(day_weight);
    daily_portfolio(t, :) = day_weight';
    
    % Cal t's return and total return
    daily_ret(t, 1) = (data(t, :)*day_weight)*(1-tc/2*sum(abs(day_weight-day_weight_o)));
    cum_ret = cum_ret * daily_ret(t, 1);
    cumprod_ret(t, 1) = cum_ret;
    
    % Adjust weight(t, :) for the transaction cost issue
    day_weight_o = day_weight.*data(t, :)'/daily_ret(t, 1);
    
    % Log information

end

% Output the cumulative return and log it.


end
