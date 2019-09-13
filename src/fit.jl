"Optimization output structure"
struct UnmarkedOpt
  coef::Array{Float64}
  vcov::Array{Float64}
  loglik::Float64
end

"Optimize a likelihood function loglik with np parameters"
function optimize_loglik(loglik, np)

  func = TwiceDifferentiable(vars -> loglik(vars), zeros(np); 
                             autodiff=:forward)

  opt = optimize(func, zeros(np), LBFGS())
  param = Optim.minimizer(opt)
  hes = NLSolversBase.hessian!(func, param)
  vcov = inv(hes)
  loglik = -Optim.minimum(opt) 
  UnmarkedOpt(param, vcov, loglik)
end

"Fitted model output type"
abstract type UnmarkedModel <: RegressionModel end

"Submodel output structure"
struct UnmarkedSubmodel <: RegressionModel
  name::Symbol
  formula::FormulaTerm
  link::Link
  coef::Array
  vcov::Array
  coefnames::Array{String}
  data::DataFrame
end

function UnmarkedSubmodel(ud::UmDesign, opt::UnmarkedOpt)
  UnmarkedSubmodel(ud.name, ud.formula, ud.link, opt.coef[ud.idx],
                   opt.vcov[ud.idx,ud.idx], ud.coefnames, ud.data)
end

#Submodel extractors
"Get detection model"
function detection(fit::UnmarkedModel)
  fit.submodels.det
end

"Get occupancy model"
function occupancy(fit::UnmarkedModel)
  fit.submodels.occ
end

"Get abundance model"
function abundance(fit::UnmarkedModel)
  fit.submodels.abun
end

"Outer constructor that re-creates design matrices from fitted model"
function UmDesign(um::UnmarkedSubmodel)
  UmDesign(um.name, um.formula, um.link, um.data)
end

"Outer constructor that re-creates design matrices with newdata"
function UmDesign(um::UnmarkedSubmodel, newdata::DataFrame)
  UmDesign(um.name, um.formula, um.link, newdata)
end

#Method for coeftable
function coeftable(um::UnmarkedSubmodel; level::Real=0.95)
  
  c = coef(um)
  se = stderror(um)
  z = abs.(c ./ se)
  pval = map(x -> 2*ccdf(Normal(0,1), x), z)
  ci = se*quantile(Normal(), (1-level)/2)
  levstr = level*100
  levstr = isinteger(levstr) ? string(Integer(levstr)) : string(levstr)
  CoefTable([c, se, z, pval, c+ci, c-ci], 
            ["Estimate", "Std.Error", "z value", "Pr(>|z|)",
             "Lower $levstr%","Upper $levstr%"],
            coefnames(um), 4, 3)
end

#Show methods
function Base.show(io::IO, um::UnmarkedSubmodel)
  println()
  println(string(um.name, ": ", um.formula))
  print(coeftable(um))
end

function Base.show(io::IO, fit::UnmarkedModel)
  
  println()
  for i = 1:length(fit.submodels)
    println(string(fit.submodels[i].name, ": ", fit.submodels[i].formula))
    println(coeftable(fit.submodels[i]))
    println()
  end
  print("AIC: ", round(aic(fit), digits=4))

end

## Misc. methods reguired for RegressionModel interface

#Return coefficient values
function coef(x::UnmarkedModel)
  return vcat(map(x -> coef(x), x.submodels)...)
end

function coef(x::UnmarkedSubmodel)
  return x.coef
end

#Return names of coefficients
function coefnames(x::UnmarkedModel)
  return vcat(map(x -> coefnames(x), x.submodels)...)
end

function coefnames(x::UnmarkedSubmodel)
  return x.coefnames
end
 
#Variance-covariance matrix
function vcov(x::UnmarkedModel)
  return x.opt.vcov
end

function vcov(x::UnmarkedSubmodel)
  return x.vcov
end

#Standard error
function stderror(x::UnmarkedModel)
  v = vcov(x)
  return sqrt.diag(v)
end

function stderror(x::UnmarkedSubmodel)
  sqrt.(diag(x.vcov))
end

#Log-likelihood
function loglikelihood(x::UnmarkedModel)
  return x.opt.loglik
end

#Deviance
function deviance(x::UnmarkedModel)
  return -2 * loglikelihood(x)
end

#Degrees of freedom
function dof(x::UnmarkedModel)
  return length(coef(x))
end

#Number of observations = number of independent sites
function nobs(x::UnmarkedModel)
  return size(x.data.y)[1]
end
