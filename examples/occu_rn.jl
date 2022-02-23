#Fit N-mixture model

using Random, Distributions, StatsFuns, Optim, NLSolversBase, ForwardDiff
using LinearAlgebra

using Unmarked

Random.seed!(123);

#Simulate data
N = 4000;
J = 5;
z = rand(Poisson(5),N);
r = 0.1;

y = Array{Int64,2}(undef, N, J);

for i = 1:N
  for j = 1:J
    p = 1 - (1-r)^z[i];
    y[i,j] = rand(Binomial(1, p));
  end
end

um = UmData(y)


@time mod = rn(@formula(λ~1), @formula(p~1), um, K=50)

#Display
truth = [log(5), logit(r)]; 
pnames = ["β[1]", "β[2]"];
cnames = ["param" "truth" "estimate" "se"];
[cnames; pnames truth coef(mod) stderror(mod)]
