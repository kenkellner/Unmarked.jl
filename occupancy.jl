using Random, Distributions, StatsFuns, Optim, NLSolversBase, ForwardDiff
using LinearAlgebra, DataFrames

Random.seed!(123);

#Simulate data
z = rand(Binomial(1,0.5),1000);
J = 5;
p = 0.4;

y = Array{Float64,1}(undef, 1000);

for i = 1:length(y)
  y[i] = rand(Binomial(J, p * z[i]));
end

struct umf
  n
  y
end

inp = umf(length(y), y)

#Fit with max likelihood
function fit_model(data::umf)

  y = data.y

  function loglik(β_raw::Array)
 
    β = logistic.(β_raw)

    ll = zeros(eltype(β_raw),length(y))
    for i = 1:data.n
      ll[i] = log(β[2] * (β[1]^y[i] * (1-β[1])^(J-y[i])) 
                  + (1-β[2]) * (y[i]==0))
    end
    return -sum(ll)
  end

  func = TwiceDifferentiable(vars -> loglik(vars), zeros(2); 
                             autodiff=:forward);

  opt = optimize(func, zeros(2));
  param = Optim.minimizer(opt);
  param = round.(param; digits=2);
  hes = NLSolversBase.hessian!(func, param);
  vcov = inv(hes);
  se = sqrt.(diag(vcov));
  pnames = ["β[1]", "β[2]"];

  df = DataFrame(Parameter=pnames, Estimate=param, SE=se)

  return df
end

fit = fit_model(inp);

#Display
truth = round.(logit.([0.4, 0.5]); digits=2); 
fit[!,:truth] = truth;
fit
