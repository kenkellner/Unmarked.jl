using Random, Distributions, StatsFuns, Optim, NLSolversBase, ForwardDiff
using LinearAlgebra, DataFrames, StatsModels

Random.seed!(123);

#Simulate data

#state covariates
elev = rand(Normal(0,1),1000);
forest = rand(Normal(0,1),1000);

site_covs = DataFrame(elev=rand(Normal(0,1),1000),
                      forest=rand(Normal(0,1),1000));

site_covs[!,:psi] = zeros(nrow(site_covs));

mm = ModelMatrix(ModelFrame(@formula(psi~elev+forest), site_covs));

psi_truth = [0, -0.5, 1.2];

psi = logistic.(mm.m * psi_truth);

z = Array{Int32}(undef,1000); 
for i = 1:1000
  z[i] = rand(Binomial(1,psi[i]),1)[1];
end

p = 0.4;
J = 5;
y = Array{Int32}(undef, 1000);
for i = 1:length(y)
  y[i] = rand(Binomial(J, p * z[i]));
end

struct umf
  y::Array{Int32}
  site_covs::DataFrame
end

inp = umf(y, site_covs);

#Fit with max likelihood
function fit_model(psi_formula, data::umf)

  y = data.y
  
  mf_psi = ModelFrame(psi_formula, data.site_covs)
  X_psi = ModelMatrix(mf_psi).m
  np_psi = size(X_psi)[2]

  np_p = 1 #temporary

  np = np_psi + np_p

  function loglik(β_raw::Array)
    
    β_psi = β_raw[1:np_psi]
    psi = logistic.(X_psi * β_psi)

    β_p = β_raw[(np_psi+1):np]
    p = logistic.(β_p)[1] #temporary

    ll = zeros(eltype(β_raw),length(y))
    for i = 1:length(y)
      ll[i] = log(psi[i] * (p^y[i] * (1-p)^(J-y[i])) 
                  + (1-psi[i]) * (y[i]==0))
    end
    return -sum(ll)
  end

  func = TwiceDifferentiable(vars -> loglik(vars), zeros(np); 
                             autodiff=:forward);

  opt = optimize(func, zeros(np));
  param = Optim.minimizer(opt);
  param = round.(param; digits=3);
  hes = NLSolversBase.hessian!(func, param);
  vcov = inv(hes);
  se = sqrt.(diag(vcov));
  pnames = [coefnames(mf_psi);"p"];

  df = DataFrame(Parameter=pnames, Estimate=param, SE=se)

  return df
end

fit = fit_model(@formula(psi~elev+forest), inp);

#Display
truth = round.([psi_truth; logit(p)]; digits=3); 
fit[!,:truth] = truth;
fit
