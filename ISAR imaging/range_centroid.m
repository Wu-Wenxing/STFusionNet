function [Xr,Rr,Vr,Ar] = range_centroid(X,f0,Df,prf)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

j = sqrt(-1);
c0 = 2.99792458e8;  % progagation velocity

[ns,nb] = size(X);  % ns: no of range cells; nb: no of Doppler bins
t0 = 1/prf;
Wr = c0/(2*Df); % range resolution window

kmn = [1:ns*nb];
tmn = kmn*t0;
fmn = f0+Df*rem(kmn-1,ns); % stepped frequency

for n=1:nb
    X_fft(:,n)= fftshift(fft(X(:,n)));
    range_track(n) = centroid1d(((abs(X_fft(:,n))).').^2);
end

for n=2:nb
    if range_track(n) < range_track(n-1)-ns/2
        range_track(n:nb) = range_track(n:nb)+ns;
    end
    if range_track(n) > range_track(n-1)+ns/2
        range_track(n:nb) = range_track(n:nb)-ns;
    end
end
T = ([1:nb]*ns*t0).';
R = (range_track*(Wr/ns))';
% curve fitting
[A,B]   = dataline(T,R);
[C,D,E] = dataquad(T,R);
R1 = A+B*T;
R2 = C+D*T+E*(T.^2);
E1 = sqrt(sum((R-R1).^2));
E2 = sqrt(sum((R-R2).^2));
if E2 <= E1/1.2, 
    re = C; 
    ve = D; 
    ae = 2*E; 
end
if E2 >  E1/1.2, 
    re = A; 
    ve = B; 
    ae = 0; 
end

% actual delay
f = f0+(ns/2)*Df;
s = Df/t0;
c = f/s;

Ar = ae;
Vr = ve-c*Ar;
Rr = re-c*Vr;
RR = Rr+Vr*T+(1/2)*Ar*(T.^2);
Rr = Rr+Wr/2;
RR = Rr+Vr*tmn+(1/2)*Ar*(tmn.^2);

vec = rot90(mat2vec(X),1);
filt = vec.*exp((-j*4*pi/c0)*(fmn.*RR));

Xr = vec2mat(filt,ns,nb);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%