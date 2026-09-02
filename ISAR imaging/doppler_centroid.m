function [Xrd,Xf,Rd,Vd,Ad] = doppler_centroid(X,f0,Df,prf,Xr,Rr,Vr,Ar)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Doppler centroid processing
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
j = sqrt(-1);
c0 = 2.99792458e8;  % progagation velocity

[ns,nb] = size(X);  % ns: no of range cells; nb: no of Doppler bins
t0 = 1/prf;
Wr = c0/(2*Df); % range resolution window

kmn = [1:ns*nb];
tmn = kmn*t0;
fmn = f0+Df*rem(kmn-1,ns); % stepped frequency

for n=1:nb-7
    Xr_fft = fftshift(fft2(Xr(:,n:n+7)));
    doppler_track(n) = centroid2d(abs(Xr_fft));
end

Tdop = ([4:nb-4]*t0*ns)';
Dtr = (doppler_track/(8*t0*ns))';

% curve fitting
[A,B]   = dataline(Tdop, Dtr);
[C,D,E] = dataquad(Tdop,Dtr);

R1 = A + B*Tdop;
R2 = C + D*Tdop + E*(Tdop.^2);
E1 = sqrt(sum((Dtr-R1).^2));
E2 = sqrt(sum((Dtr-R2).^2));
if E2 <= E1/1.2, 
    r1 = C; 
    v1 = D; 
    a1 = 2*E; 
end
if E2 >  E1/1.2, 
    r1 = A; 
    v1 = B; 
    a1 = 0; 
end

fc = f0+(nb/2)*Df;
Lc = c0/fc;
Rd = Rr;
Vd = (Lc/2)*r1+Vr;
Ad = (Lc/2)*v1+Ar;

R2 = Rd+Vd*tmn+(1/2)*Ad*(tmn.^2);
vec = rot90(mat2vec(X),1);
filt = vec.*exp((-j*4*pi/c0)*(fmn.*R2));

Xrd = vec2mat(filt,ns,nb);

%
% Doppler refinement
%
for n=1:nb
    Xrd_cfft(:,n) = fftshift(fft(Xrd(:,n)));
    range_track(n) = centroid1d(((abs(Xrd_cfft(:,n))).').^2);
end
for m=1:ns
    Xrd_rfft(m,:) = fftshift(fft(Xrd(m,:)));
end

Y = range_track;

for n=2:nb
    if Y(n) < Y(n-1)-ns/2
        Y(n:nb) = Y(n:nb)+ns;
    end
    if Y(n) > Y(n-1)+ns/2
        Y(n:nb) = Y(n:nb)-ns;
    end
end

T = ([1:nb]*t0*ns)';
R = (Y*(Wr/ns))';

[A,B] = dataline(T,R);
%R1 = A + B*T;

r1 = A;
v1 = B;
a1 = 0;

Rf = Rd;
Vf = Vd+v1;
Af = Ad+a1;

R = Rf+Vf*tmn+(1/2)*Af*(tmn.^2);

vec = rot90(mat2vec(X),1);
filt = vec.*exp((-j*4*pi/c0)*(fmn.*R));
Xf = vec2mat(filt,ns,nb);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%