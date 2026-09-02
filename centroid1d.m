function [tc] = centroid1d(x)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% calculate 1-D centroid location
%
% x: 1-D time seriers with even length
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

sz = length(x);
sz2 = sz/2;
x_sum = sum(x);
t = 1:sz; % time samples

xc = circshift(x,[1 -sz2]); % circular shift right by sz2

ttx = sum(t.*x)/x_sum;
ttxc = sum(t.*xc)/x_sum;

Ittx = sum(((t-ttx).^2).*x);
Ittxc = sum(((t-ttxc).^2).*xc);

if Ittx <= Ittxc; 
    tc = ttx;      
end
if Ittx >  Ittxc; 
    tc = ttxc - sz2; 
end

if tc <= 0; 
    tc = tc + 2*sz2; 
end
if tc > sz; 
    tc = tc - 2*sz2; 
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

