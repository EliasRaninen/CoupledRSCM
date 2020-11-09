function [RSCM,iRSCM,al,be,MSEhat,MSEhatmin] = rscmpool(params,average,compute_inverse)
%%RSCMPOOL computes the regularized sample covariance matrix estimators
% for each class k, the function computes
%   RSCM{k} = al(k)*C + (1-al(k))*I_C, where
%   C = be(k)*SCM{k} + (1-be(k))*S, and
%   I_C = trace(C)*eye(p)/p, and S denotes the pooled SCM.
%
% Input:
%   params  - parameters computed by function estimate_parameters.m
%   average - true/false, if true then al and be are averaged over the
%             classes (default setting is false).
%   compute_inverse - true/false. If true, then the inverses of the
%                     covariance estimates are computed (default setting is
%                     false).
%
% Output:
%   RSCM - cell of regularized SCM estimators for each class
%   iRSCM - cell of inverses of RSCMs
%   al - estimated regularization parameters
%   be - estimated regularization parameters
%   MSEhat(i,j,k) - estimated MSE of class k using regularization
%       parameters (al(i),be(j)), where the grid of al and be is created
%       with linspace(0,1,21).
%   MSEhatmin - estimated MSE for the chosen alpha and beta.
%
% Example:
%
% [RSCM,iRSCM,al,be,MSEhat,MSEhatmin] = rscmpool(params,average)
%
% by Elias Raninen 2020

% average the computed alpha and beta regularization parameters across classes
if nargin < 2
    average = false;
end
if nargin < 3
    compute_inverse = false;
end
% compute the rscm
compute_rscm = true;


p   = params.p; % dimension
K   = params.K; % number of classes

%% Terms for computing the coefficients of the MSE polynomial
S2 = params.Etr_S2;
IS2 = params.EtrS_2/p;
SI2 = S2 - IS2;

Sk2 = diag(params.EtrSiSj);
ISk2 = diag(params.EtrSitrSj)/p;
SkI2 = Sk2 - ISk2;

SkS   = params.EtrSkS;
ISkIS = params.EtrSktrS/p;
SkISI = SkS - ISkIS;

SCk = params.EtrCkS;
ISCk = params.EtrCktrS/p;
SICk = SCk - ISCk;

Ck2 = diag(params.trCiCj);
SkCk = Ck2;
ISkCk = diag(params.trCitrCj)/p;
SkICk = SkCk - ISkCk;

%% Coefficients of MSE polynomial
C22 = SkI2 + SI2 - 2*SkISI;
C21 = 2*(SkISI - SI2);
%C12 = 0;
C20 = SI2;
C02 = ISk2 + IS2 - 2*ISkIS;
C11 = -2*(SkICk - SICk);
C10 = -2*SICk;
C01 = 2*(ISkIS - ISkCk - IS2 + ISCk);
C00 = IS2 + Ck2 - 2*ISCk;

%% MSE polynomial
MSEfun = @(a,b) a.^2.*b.^2.*C22 + a.^2.*b.*C21 ...
    + a.^2.*C20 + b.^2.*C02 ...
    + a.*b.*C11 + a.*C10 + b.*C01 + C00;

% zero gradient equations
%nabla_alpha = @(a,b) 2*a*b^2.*C22 + 2*a*b*C21 + 2*a*C20 + b*C11 + C10;
%nabla_beta = @(a,b) 2*a^2*b*C22 + a^2*C21 + 2*b*C02 + a*C11 + C01;

% optimize alpha given beta
opt_al = @(b) -1/2*(b.*C11+C10)./(b.^2.*C22+b.*C21+C20);
% optimize beta given alpha
opt_be = @(a) -1/2*(a.^2.*C21+a.*C11+C01)./(a.^2.*C22+C02);

%% Make a grid of alpha and beta and choose the best for each class
arrlen  = 21;
alpha_arr = linspace(0,1,arrlen).';
beta_arr = linspace(0,1,arrlen).';

MSEhat = nan(arrlen,arrlen,K);
for i=1:arrlen
    al = alpha_arr(i);
    for j=1:arrlen
        be = beta_arr(j);
        MSEhat(i,j,:) = MSEfun(al,be);
    end
end

al = nan(K,1);
be = nan(K,1);
%mins = min(MSEhat,[],[1 2]); % minimum mse of each class
mins = nan(K,1);
for k=1:K
    tmp = MSEhat(:,:,k);
    mins(k) = min(tmp(:));
    [i,j] = find(MSEhat(:,:,k) == mins(k));
    al(k) = alpha_arr(i);
    be(k) = beta_arr(j);
end

%% Make a grid of alpha then optimize beta given alpha and choose best
% MSEalgrid = nan(arrlen,K);
% for i=1:arrlen
%     al = alpha_arr(i);
%     MSEalgrid(i,:) = MSEfun(al,opt_be(al));
% end
% [~,j] = min(MSEalgrid);
% al    = alpha_arr(j);
% be    = min(1,max(0,opt_be(al)));

%% Find local minima given initial alpha and beta
iterMAX = 1000;
for iter=1:iterMAX
    al0 = al;
    be0 = be;
    al = min(1,max(0,opt_al(be)));
    be = min(1,max(0,opt_be(al)));
    crit = norm([al0; be0] - [al; be],'fro')/norm([al0; be0],'fro');
    if crit < 1e-8
        break;
    end
end
if iter == iterMAX
    fprintf('RSCMPOOL.M: Slow convergence.');
end

%% Average coefficient over classes
if average
    al = ones(K,1)*mean(al);
    be = ones(K,1)*mean(be);
end

%% Estimated MSE at chosen alpha and beta
MSEhatmin = MSEfun(al,be);

%% Compute RSCM and its inverse using solved alpha and beta

if compute_rscm == true
    RSCM = cell(K,1);
    iRSCM = cell(K,1);
    
    SCM = params.SCM;
    Sp  = params.S;
    
    X = cell2mat(params.Xc);
    n = params.n;
    PI = params.PI;
    
    for k=1:K
        Sb = be(k)*SCM{k} + (1-be(k))*Sp;
        RSCM{k} = al(k)*Sb + (1-al(k))*(trace(Sb)/p)*eye(p);
        
        if compute_inverse
            if and(al(k) < 1, sum(n) < p) % faster computation of inverse
                c = cell(K,1);
                for j=1:K
                    if j==k
                        c{j} = sqrt((be(k)+(1-be(k))*PI(j))/(n(j)-1))*ones(n(j),1);
                    else
                        c{j} = sqrt((1-be(k))*PI(j)/(n(j)-1))*ones(n(j),1);
                    end
                end
                C = cell2mat(c);
                A = C.*X;
                om = (1-al(k))*trace(Sb)/p;
                alom = al(k)/om;
                iRSCM{k} = (1/om)*(eye(p)-alom*A'*((eye(sum(n))+alom*(A*A'))\A));
            else
                iRSCM{k} = RSCM{k} \ eye(p);
            end
        end
    end
end