function corr = calcPearsonCorr2(a, b)
% When only correlation value is needed, this function is about 30% faster
% than the Matlab build-in function corr(x,y)
% Oct. 2014, Chau-Wai Wong
% Apr. 2018 Runze Liu
[m,n] = size(a);
a = reshape(a, m*n, 1);
b = reshape(b, m*n, 1);
a1 = a - mean(a);
b1 = b - mean(b);

var_a = sum(a1.^2);
var_b = sum(b1.^2);

corr = a1'*b1 / sqrt(var_a*var_b);