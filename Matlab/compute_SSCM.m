function SSCM = compute_SSCM(X,n,p)
%% COMPUTE_SSCM computes the sample spatial sign covariance matrix.
% n is the number of samples, p is the dimension.

assert(isequal(size(X),[n,p]));

% compute spatial median
mu = spatialmedian(X,n,p);
assert(isequal(size(mu),[1 p]));
% center by the spatial median
Xc = X - repmat(mu,n,1);
% compute the norm of each sample
normXc = sqrt(sum(Xc.^2,2));
% divide each sample by the its norm (take them to the unit sphere)
U  = Xc./repmat(normXc,1,p);
% compute sample covariance of normalized samples
SSCM = U'*U/n; % normalization by n
