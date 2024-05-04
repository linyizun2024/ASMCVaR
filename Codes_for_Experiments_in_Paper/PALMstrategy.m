function [Param,CW,all_w,t,runout] = PALMstrategy(Param,data)

runout = 0;
fullR = data-1;
[fullT,d] = size(fullR);
T_end = fullT;
all_w = ones(d,fullT)/d; %d*T?
tmp = zeros(1+Param.winsize,fullT);

CW = zeros(T_end,1);  %Cumulative Wealth
S = 1;

% 
for t = 1:T_end
    if t>5 %equally-weighted portfolio
        vech1 = [zeros(d,1);1;1/((1-Param.c)*Param.winsize)*ones(Param.winsize,1)];
        vecq = [zeros(2*Param.winsize+d,1);1;-1];
        if t<=Param.winsize
            win_start = 1;
            vech1 = [zeros(d,1);1;1/((1-Param.c)*(t-1))*ones(t-1,1)];
            vech2 = [mean(fullR(1:t,:),1),zeros(1,t)]';
            vecq  = [zeros(2*(t-1)+d,1);1;-1];
        else
            win_start = t-Param.winsize;
            vech2 = [mean(fullR(win_start:t-1,:),1),zeros( 1,Param.winsize+1 )]';
        end
        win_end = t-1;
        n = win_end-win_start+1; %?

        matR = fullR(win_start:win_end,:);
        matQ = [[[[matR,ones(n,1)];zeros(n,d+1)],[eye(n);eye(n)]];[[eye(d),zeros(d,n+1)];[ones(1,d),zeros(1,n+1)];[-ones(1,d),zeros(1,n+1)]]];
        
        v_onlyw = [1/d*ones(d,1);0;1/n*ones(n,1)];

        Param.lambda = sqrt(n)*(vech1'*v_onlyw)/ (vech2'*v_onlyw-Param.rho)^2;


        Param.gammacoe = 0.00001; 
        Param.Lip_1 = 2*Param.lambda*norm(vech2,2)^2+1/Param.gammacoe;
        Param.Lip_2 = 1/Param.gammacoe;
        
        Param.beta_1 = 0.99/(Param.Lip_1);
        Param.beta_2 = 0.99/(Param.Lip_2);
       
        Param.alphacoe = Param.beta_2 / Param.gammacoe; 
        Param.omegacoe = Param.beta_1; 
        
        Param.theta = 1.99/(norm(matQ,2))^2;  
        
        [w,w_cons] = FPPAforPALM(Param,matR,matQ,vech1,vech2,vecq,n);

        all_w(:,t) = w;
    end
    
    %Adjust portfolio for the transaction cost issue
    if t==1
        daily_port_o = zeros(d,1);
    else
        daily_port_o = all_w(:,t-1).*data(t-1, :)'/(data(t-1, :)*all_w(:,t-1));
    end
    
    S = S*data(t,:)*all_w(:,t)*(1-Param.trancost/2*sum(abs(all_w(:,t)-daily_port_o)));
    CW(t) = S;
    
    if isnan(CW(t))
        fprintf('CW(t) is NaN. Skipping to the next iteration.\n');
        runout = 1;
            break; 
    end

end
end