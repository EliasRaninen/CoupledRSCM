% By E. Raninen 2020

clear; clc;
rng(123);

%% Define four classes (populations)
% number of classes
K = 4;
% number of samples
n = [100 100 100 100].';
% dimension
p = 200;
% degrees of freedom of multivariate t data
dof = [12 8 12 8];
    
% function for generating AR1
AR1cov = @(r,p) r.^abs(repmat(1:p,p,1)-repmat(1:p,p,1)');
% function for generating CS
CScov  = @(r,p) r*(ones(p)-eye(p)) + eye(p);

% the population covariance matrices
rho = [0.6 0.6 0.1 0.1];
trueCovarianceMatrices{1} = AR1cov(rho(1),p);
trueCovarianceMatrices{2} = AR1cov(rho(2),p);
trueCovarianceMatrices{3} = CScov(rho(3),p);
trueCovarianceMatrices{4} = CScov(rho(4),p);

% the means of the classes
trueMeans{1} = randn(p,1);
trueMeans{2} = randn(p,1);
trueMeans{3} = randn(p,1);
trueMeans{4} = randn(p,1);

%% Monte Carlo loop
% number of Monte Carlos
nmc = 4000;

NSE1 = nan(nmc,K); % normalized squared error (NSE) for POLY
NSE1ave = nan(nmc,K);
NSE2 = nan(nmc,K); % for POLYs
NSE2ave = nan(nmc,K);

% function for computing normalized MSE
NSE = @(A,k) norm(A-trueCovarianceMatrices{k},'fro')^2/norm(trueCovarianceMatrices{k},'fro')^2;

for mc=1:nmc
    %% Generate multivariate t samples from classes

    % To generate multivariate t with covariance Sig, we need to divide
    % Sig with the variance dof/(dof-2).
    variance = dof./(dof-2);
    
    dataFromClasses = cell(K,1);
    for k=1:K
        Sig = trueCovarianceMatrices{k};
        % normal distributed N(0,Sig/variance)
        N = randn(n(k),p)*sqrtm(Sig/variance(k));
        % multivariate t with covariance matrix Sig
        X = N ./ repmat(sqrt(chi2rnd(dof(k), n(k), 1)/dof(k)), 1, p);
        % add the mean
        dataFromClasses{k} = X + repmat(trueMeans{k}.', n(k), 1);
    end
    
    %% Estimate covariance matrices from the data
    params = estimate_parameters(dataFromClasses);
    
    % proposed method
    POLY = rscmpool(params);
    % proposed method using averaging of regularization parameters
    POLYave = rscmpool(params,true);
    % streamlined analytical version with identity shrinkage towards tr(Spool)/p*I
    POLYs = rscmpools(params,'S');
    % streamlined analytical version using averaging of regularization parameters
    POLYsave = rscmpools(params,'S',true);
    
    %% Compute normalized squared error NSE
    for k=1:K
        NSE1(mc,k) = NSE(POLY{k},k);
        NSE1ave(mc,k) = NSE(POLYave{k},k);
        NSE2(mc,k) = NSE(POLYs{k},k);
        NSE2ave(mc,k) = NSE(POLYsave{k},k);
    end
    
    if mod(mc,20)==0; fprintf('.'); end
end
fprintf('\n');
%% NMSE

% Average results over Monte Carlos
NMSE1 = [mean(NSE1) mean(sum(NSE1,2))]*10;
STD1 = [std(NSE1) std(sum(NSE1,2))]*10;

NMSE2 = [mean(NSE2) mean(sum(NSE2,2))]*10;
STD2 = [std(NSE2) std(sum(NSE2,2))]*10;

NMSE1ave = [mean(NSE1ave) mean(sum(NSE1ave,2))]*10;
STD1ave = [std(NSE1ave) std(sum(NSE1ave,2))]*10;

NMSE2ave = [mean(NSE2ave) mean(sum(NSE2ave,2))]*10;
STD2ave = [std(NSE2ave) std(sum(NSE2ave,2))]*10;

disp('Normalized MSE (x10) and standard deviation (x10) for the four classes.');
fprintf('(averaged over %d Monte Carlo trials.)\n',nmc);
% Table for NMSE
T = splitvars(table(round([NMSE1;NMSE2;NMSE1ave;NMSE2ave],2)));
T.Properties.VariableNames = {'Class1','Class2','Class3','Class4','sum'};
T.Properties.RowNames = {'POLY NMSE:','POLYs NMSE:','POLYave NMSE:','POLYsave NMSE:'};
disp(T);

% Table for standard deviation
Tstd = splitvars(table(round([STD1;STD2;STD1ave;STD2ave],3)));
Tstd.Properties.VariableNames = {'Class1','Class2','Class3','Class4','sum'};
Tstd.Properties.RowNames = {'POLY STD:','POLYs STD:','POLYave STD:','POLYsave STD:'};
disp(Tstd)