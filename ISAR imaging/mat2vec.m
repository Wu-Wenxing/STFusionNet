function [vector] = mat2vec(matrix)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% mat2vec   -  converts matrix into a vector.

%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[N,M] = size(matrix);
vector = zeros(N.*M,1);

for k = 1:N.*M;
   vector(k) = matrix(k);
end
