function [RSCM,iRSCM,al,be,MSEhat,MSEhatmin] = rscmpools(params,T,average,compute_inverse,compute_mse_grid)
%%RSCMPOOLS computes the regularized sample covariance matrix estimators.
% For each class k, the function computes
%   RSCM{k} = al(k)*C + (1-al(k))*T,
%   T is either SCM{k} or S, where S denotes the pooled SCM.
%
% Input:
%   params  - parameters computed by function estimate_parameters.m
%   T       - either 'Sk' or 'S', where Sk denotes the class SCM and S the pooled SCM
%   average - true/false, if true then al and be are averaged over the classes
%   compute_inverse - true/false. If true, then the inverses of the
%                     covariance estimates are computed (default setting is
%                     false).
%   compute_mse_grid - true/false. If true, then estimates of the MSE are
%                      computed for a grid of alpha and beta values.
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
% [RSCM,iRSCM,al,be,MSEhat,MSEhatmin] = rscmpools(params,T,average,compute_inverse,compute_mse_grid)
%
% by Elias Raninen 2020

% average the computed alpha and beta regularization parameters across classes
if nargin < 3
    average = false;
end
if nargin < 4
    compute_inverse = false;
end
if nargin < 5
    compute_mse_grid = false;
end
% compute the rscm
compute_rscm = true;

p   = params.p; % dimension
K   = params.K; % number of classes

%% Terms for computing the coefficients of the MSE polynomial
S2 = params.Etr_S2;
IS2 = params.EtrS_2/p;

Sk2 = diag(params.EtrSiSj);
ISk2 = diag(params.EtrSitrSj)/p;

SkS   = params.EtrSkS;
ISkIS = params.EtrSktrS/p;

SCk = params.EtrCkS;
ISCk = params.EtrCktrS/p;

Ck2 = diag(params.trCiCj);
SkCk = Ck2;
ISkCk = diag(params.trCitrCj)/p;

%%
if isequal(T,'Sk')
    ISkIT = ISk2;
    ISIT = ISkIS;
    IT2 = ISk2;
    ITCk = ISkCk;
elseif isequal(T,'S')
    ISkIT = ISkIS;
    ISIT = IS2;
    IT2 = IS2;
    ITCk = ISCk;
end

%% Coefficients of MSE polynomial
B22 = Sk2 + S2 - 2*SkS;
B21 = 2*(SkS - S2 - ISkIT + ISIT);
B20 = S2 + IT2 - 2*ISIT;
B11 = 2*(ISkIT - SkCk - ISIT + SCk);
B10 = 2*(ISIT - SCk - IT2 + ITCk);
B00 = IT2 + Ck2 - 2*ITCk;

%% MSE polynomial
MSEfun = @(a,b) a.^2.*b.^2.*B22 + a.^2.*b.*B21 ...
    + a.^2.*B20 + a.*b.*B11 + a.*B10 + B00;

%% Compute MSE grid
if compute_mse_grid
    arrlen = 21;
    MSEhat = nan(arrlen,arrlen,K);
    alpha_arr = linspace(0,1,arrlen).';
    beta_arr  = linspace(0,1,arrlen).';
    for i=1:arrlen
        al_i = alpha_arr(i);
        for j=1:arrlen
            be_j = beta_arr(j);
            MSEhat(i,j,:) = MSEfun(al_i,be_j);
        end
    end
end
%% alpha and beta candidates
alc = nan(5,K);
bec = nan(5,K);

% when (alpha,beta) is in the interior of [0,1] X [0,1]
alc(1,:) = (2*B10.*B22-B11.*B21)./(B21.^2-4*B20.*B22);
bec(1,:) = (2*B11.*B20-B10.*B21)./(2*B10.*B22-B11.*B21);

% at the borders of [0,1] X [0,1]
alc(2,:) = 0;
bec(2,:) = 0; % when al = 0, the estimator doesn't depend on be

alc(3,:)  = 1;
bec(3,:) = (-1/2)*(B21+B11)./B22; % al=1

alc(4,:) = (-1/2)*B10./B20; % be=0
bec(4,:) = 0;

alc(5,:) = (-1/2)*(B11+B10)./(B22 + B21 + B20); % be=1
bec(5,:) = 1;

% restrict tuning parameter candidates to [0,1]
bec = min(1,max(0,bec));
alc = min(1,max(0,alc));

% estimate mse of different alpha and beta candidates
mse_alc_bec = nan(5,K);
for ii=1:5
    mse_alc_bec(ii,:) = MSEfun(alc(ii,:)',bec(ii,:)');
end

% choose the one which has lowest estimated mse
al = nan(K,1);
be = nan(K,1);
for k=1:K
    [~,idx] = min(mse_alc_bec(:,k));
    al(k) = alc(idx,k);
    be(k) = bec(idx,k);
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
        if isequal(T,'Sk')
            RSCM{k} = al(k)*Sb + (1-al(k))*params.eta(k)*eye(p);
        elseif isequal(T,'S')
            RSCM{k} = al(k)*Sb + (1-al(k))*(trace(Sp)/p)*eye(p);
        end
        if compute_inverse
            if and(al(k) < 1, sum(params.n) < p) % faster computation of inverse
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
                if isequal(T,'Sk')
                    om = (1-al(k))*trace(SCM{k})/p;
                elseif isequal(T,'S')
                    om = (1-al(k))*trace(Sp)/p;
                end
                alom = al(k)/om;
                iRSCM{k} = (1/om)*(eye(p)-alom*A'*((eye(sum(n))+alom*(A*A'))\A));
            else
                iRSCM{k} = RSCM{k} \ eye(p);
            end
        end
    end
end
