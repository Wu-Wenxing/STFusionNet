function Xf = refine(X,f0,Df,prf,Xrd,Rd,Vd,Ad)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% final refining processing

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

j = sqrt(-1);
c0 = 3e8;

[ns,nb] = size(X);
t0 = 1/prf;
Wr = c0/(2*Df);

kmn = [1:ns*nb];
tmn = kmn*t0;
fmn = f0+Df*rem(kmn-1,ns);

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

r1 = A;
v1 = B;
a1 = 0;

Rf = Rd+r1;
Vf = Vd+v1;
Af = Ad+a1;

R = Rf+Vf*tmn+(1/2)*Af*(tmn.^2);

vec = rot90(mat2vec(X),1);
filt = vec.*exp((-j*4*pi/c0)*(fmn.*R));
Xf = vec2mat(filt,ns,nb);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%