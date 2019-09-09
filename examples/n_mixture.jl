#Fit N-mixture model

using Random, Distributions, StatsFuns, Optim, NLSolversBase, ForwardDiff
using LinearAlgebra

Random.seed!(123);

#Simulate data
N = 100;
J = 5;
z = rand(Poisson(10),N);
p = 0.4;

y = Array{Int64,2}(undef, N, J);

for i = 1:N
  for j = 1:J
    y[i,j] = rand(Binomial(z[i], p));
  end
end

#Population size values to marginalize over
K = maximum(y) + 20

#Estimate

function loglik(β_raw::Array)
  
  λ = exp(β_raw[1])
  p = logistic(β_raw[2])
   
  ll = zeros(eltype(β_raw),N)
  for i = 1:N
    fg = zeros(eltype(β_raw),K+1)
    for k = 0:K
      if maximum(y[i,]) > k
        continue
      end
      fk = pdf(Poisson(λ), k)
      gk = 1.0
      for j = 1:J
        gk *= pdf(Binomial(k, p), y[i,j])
      end
      fg[k+1] = fk * gk
    end
    ll[i] = log(sum(fg))
  end

  return -sum(ll)
end
      
func = TwiceDifferentiable(vars -> loglik(vars), zeros(2); autodiff=:forward);

opt = optimize(func, zeros(2));
param = Optim.minimizer(opt);
hes = NLSolversBase.hessian!(func, param);
vcov = inv(hes);
se = sqrt.(diag(vcov));

#Display
truth = [log(10), logit(p)]; 
pnames = ["β[1]", "β[2]"];
cnames = ["param" "truth" "estimate" "se"];
[cnames; pnames truth param se]
