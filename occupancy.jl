using Random, Distributions, StatsFuns, Optim, NLSolversBase, ForwardDiff
using LinearAlgebra

Random.seed!(123);

#Simulate data
z = rand(Binomial(1,0.5),100);
J = 5;
p = 0.4;

y = Array{Float64,1}(undef, 100);

for i = 1:length(y)
  y[i] = rand(Binomial(J, p * z[i]));
end

#Estimate

function loglik(β_raw::Array)
 
  β = map.(logistic, β_raw)

  ll = zeros(eltype(β_raw),length(y))
  for i = 1:length(y)
    ll[i] = log(β[2] * (β[1]^y[i] * (1-β[1])^(J-y[i])) + (1-β[2]) * (y[i]==0))
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
truth = [-0.41, 0.0]; 
pnames = ["β[1]", "β[2]"];
cnames = ["param" "truth" "estimate" "se"];
[cnames; pnames truth param se]
