"Optimization output structure"
struct UmOpt
  coef::Array{Float64}
  se::Array{Float64}
  vcov::Array{Float64}
  AIC::Float64
end

"Optimize a likelihood function loglik with np parameters"
function optimize_loglik(loglik, np)

  func = TwiceDifferentiable(vars -> loglik(vars), zeros(np); 
                             autodiff=:forward)

  opt = optimize(func, zeros(np), LBFGS())
  param = Optim.minimizer(opt)
  hes = NLSolversBase.hessian!(func, param)
  vcov = inv(hes)
  se = sqrt.(diag(vcov))
  AIC = 2 * np - 2 * -Optim.minimum(opt)

  UmOpt(param, se, vcov, AIC)
end

"Fitted model output structure"
struct UmFit
  coef::Array
  se::Array
  vcov::Array
  AIC::Float64
  coef_names::Array
  model_names::Array
  param_names::Array
  inds::Array
  formulas::Array
end

"Build table of coefficients and related stats for given model"
function coeftable(fit::UmFit, model::String)
  inds = fit.inds[get_index(model, fit.model_names)]
  coef = fit.coef[inds]
  se = fit.se[inds]
  z = abs.(coef./se)
  pval = map(x -> 2*ccdf(Normal(0,1), x), z)

  CoefTable([coef, se, z, pval], 
            ["Estimate", "Std.Error", "z value", "Pr(>|z|)"],
            fit.coef_names[inds], 4, 3)
end

"Show function for UmFit"
function Base.show(io::IO, fit::UmFit)
  
  println()
  for i = 1:length(fit.model_names)
    println(string(fit.model_names[i], ": ", fit.formulas[i]))
    println(coeftable(fit, fit.model_names[i]))
    println()
  end
  print("AIC: ", round(fit.AIC, digits=4))

end
