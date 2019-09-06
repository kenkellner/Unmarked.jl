using DataFrames: DataFrame
using StatsModels: ModelMatrix, ModelFrame, @formula, Term, diag
using StatsFuns: logistic
using Distributions: ccdf
using StatsBase: CoefTable
using Optim, NLSolversBase, ForwardDiff

struct umf
  y::Array{Int}
  site_covs::DataFrame
end

#Fit with max likelihood
function fit_model(psi_formula, data::umf)

  y = data.y

  gd = get_design([psi_formula], [data.site_covs])
  
  #temporary
  np_p = 1
  psi_ind = gd.inds[1]
  np_psi = length(psi_ind)
  X_psi = gd.mats[1]
  np = np_psi + np_p
  pnames = [gd.coefs[1];"(Intercept)"]
  modnames = ["Occupancy", "Detection"]
  inds = [gd.inds, [np:np]]

  function loglik(β_raw::Array)
    
    β_psi = β_raw[psi_ind]
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

  opt = optimize_loglik(loglik, np)

  UmFit(opt.coef, opt.se, opt.vcov, pnames, modnames, inds)

end
