function [v] = ProxlmI(u,d,m)

    tmp = u(1:d);
    [Val,Index] = maxk(tmp,m);
    v = zeros(length(u),1);
    for i = 1:length(Index)
        v(Index(i)) = Val(i);
    end
        
end