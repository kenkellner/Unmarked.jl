import Base.show

"Optimization output structure"
struct ParamOpt
  coef::Array{Float64}
  se::Array{Float64}
  vcov::Array{Float64}
end

"Optimize a likelihood function loglik with np parameters"
function optimize_loglik(loglik, np)

  func = TwiceDifferentiable(vars -> loglik(vars), zeros(np); 
                             autodiff=:forward)

  opt = optimize(func, zeros(np), BFGS())
  param = Optim.minimizer(opt)
  hes = NLSolversBase.hessian!(func, param)
  vcov = inv(hes)
  se = sqrt.(diag(vcov))

  ParamOpt(param, se, vcov)
end

"Fitted model output structure"
struct UmFit
  coef::Array
  se::Array
  vcov::Array
  coef_names::Array
  model_names::Array
  inds::Array
end

"Build table of coefficients and related stats for given model"
function coeftable(fit::UmFit, model::String)
  inds = fit.inds[findall(fit.model_names .== model)[1]][1]
  coef = fit.coef[inds]
  se = fit.se[inds]
  z = abs.(coef./se)
  pval = map(x -> 2*ccdf(Normal(0,1), x), z)

  CoefTable([coef, se, z, pval], ["Estimate", "SE", "z", "P(>|z|)"],
            fit.coef_names[inds], 4, 3)
end

"Show function for UmFit"
function Base.show(io::IO, fit::UmFit)
  
  for i = 1:length(fit.model_names)
    println(fit.model_names[i])
    println(coeftable(fit, fit.model_names[i]))
  end

end
