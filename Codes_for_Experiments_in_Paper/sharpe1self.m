function [ sharpe_ratio ] = sharpe1self( strategy_CW )
strategy_CW=tick2ret(strategy_CW);

sharpe_ratio=sharpe(strategy_CW);
end

