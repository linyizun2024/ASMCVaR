function [w,w_cons] = FPPAforPALM(Param,matR,matQ,vech1,vech2,vecq,n)

%% Initialization
[T,d] = size(matR);
v = [1/d*ones(d,1);zeros(1+n,1)];
y = [1/d*ones(d,1)];
RE1 = zeros(Param.MaxIter1+1,1);  % Relative Error: ||v^(k+1)-v^(k)||_2 / ||v^(k)||_2
RE1(1) = inf;
k1 = 1;
w_cons = 0;

%% Fixed-point iteration
while k1<=Param.MaxIter1 && RE1(k1)>Param.tol_2
    RE2 = zeros(Param.MaxIter2+1,1);
    RE2(1) = inf;
    k2 = 1;
    v_pre = v;
    y_pre = y;
    
% vector v
    df = vech1 + 2*Param.lambda*vech2*(vech2'*v_pre-Param.rho);
    p = v_pre - Param.omegacoe * (df + (1/Param.gammacoe) * ( [v_pre(1:d);zeros(1+n,1)]- [y_pre;zeros(1+n,1)]));
    

    x = matQ * p;
    while k2 <= Param.MaxIter2 && RE2(k2) > Param.tol_1 
        x_pre = x;
        tmp = matQ * p + x_pre - Param.theta * (matQ*(matQ'*x_pre));
        x = (1-Param.kappa) * x_pre + Param.kappa * tmp -Param.kappa*max(tmp,vecq);
        RE2(k2+1) = norm(x-x_pre,2)/norm(x_pre,2);
        k2 = k2+1;
    end
    v = p - Param.theta * (matQ'*x);
    
    % vector y
    tmp2 = Param.alphacoe * v(1:d) + (1- Param.alphacoe) * y_pre;
    y = ProxlmI(tmp2,d,Param.m);
 
    RE1(k1+1) = norm(v-v_pre,2)/norm(v_pre,2);
    k1 = k1+1;
end

w = v(1:d);


if max(abs(w)) > 2
    w_cons = 1;
end
w(w < 0.002) = 0;
if sum(w)>1
    w = simplex_projection_selfnorm2(w, 1);
else
    less_num = 1 - sum(w);
    [~, indices] = maxk(w,floor(Param.m/5));
    maxElements = w(indices);
    scalingFactor = less_num / length(indices);
    w(indices) = w(indices)+scalingFactor*ones(length(indices),1);
end
    
end