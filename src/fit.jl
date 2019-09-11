"Optimization output structure"
struct UmOpt
  coef::Array{Float64}
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
  AIC = 2 * np - 2 * -Optim.minimum(opt)

  UmOpt(param, vcov, AIC)
end

"Fitted model output type"
abstract type UmFit end

"Submodel output structure"
struct UmModel
  name::Symbol
  formula::FormulaTerm
  link::Link
  coef::Array
  vcov::Array
  coefnames::Array{String}
  data::DataFrame
end

function UmModel(ud::UmDesign, opt::UmOpt)
  UmModel(ud.name, ud.formula, ud.link, opt.coef[ud.idx],
          opt.vcov[ud.idx,ud.idx], ud.coefnames, ud.data)
end

"Outer constructor that re-creates design matrices from fitted model"
function UmDesign(um::UmModel)
  UmDesign(um.name, um.formula, um.link, um.data)
end

"Outer constructor that re-creates design matrices with newdata"
function UmDesign(um::UmModel, newdata::DataFrame)
  UmDesign(um.name, um.formula, um.link, newdata)
end

"Get array with all coefficients"
function coef(x::UmFit)
  vcat(map(x -> x.coef, x.models)...)
end

"Calculate standard errors from vcov matrix"
function SE(x::UmModel)
  sqrt.(diag(x.vcov))
end

"Generate coeftable for UmModel"
function coeftable(um::UmModel)
  
  se = SE(um)
  z = abs.(um.coef ./ se)
  pval = map(x -> 2*ccdf(Normal(0,1), x), z)

  CoefTable([um.coef, se, z, pval], 
            ["Estimate", "Std.Error", "z value", "Pr(>|z|)"],
            um.coefnames, 4, 3)
end

"Show function for UmModel"
function Base.show(io::IO, um::UmModel)
  println()
  println(string(um.name, ": ", um.formula))
  print(coeftable(um))
end

"Show function for UmFit"
function Base.show(io::IO, fit::UmFit)
  
  println()
  for i = 1:length(fit.models)
    println(string(fit.models[i].name, ": ", fit.models[i].formula))
    println(coeftable(fit.models[i]))
    println()
  end
  print("AIC: ", round(fit.opt.AIC, digits=4))

end
