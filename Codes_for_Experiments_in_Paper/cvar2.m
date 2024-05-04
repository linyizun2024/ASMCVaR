function [ v,cv ] = cvar2( CW )

R=tick2ret([1;CW]);
c=0.95;
T=length(R);

h = [1;1/((1-c)*T)*ones(T,1)];
Q = -[ones(T,1),eye(T);zeros(T,1),eye(T)];
q = -[-R;zeros(T,1)];

[v,cv] = linprog(h,Q,q) ;

end

