function w = simplex_projection_selfnorm2(v, b)


while(max(abs(v))>1e6)
v=v/10;
end

u = sort(v,'descend');

sv = cumsum(u);
rho = find(u > (sv - b) ./ (1:length(u))', 1, 'last');
theta = (sv(rho) - b) / rho;
w = max(v - theta, 0);
end