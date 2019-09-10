using Random: seed!, rand
using Distributions: Normal, Binomial
using DataFrames: DataFrame, nrow
using StatsModels: @formula, ModelFrame, ModelMatrix
using StatsFuns: logistic, logit

using Unmarked

seed!(123);

#Simulate data

N = 1000;
J = 5;

#state covariates
site_covs = DataFrame(elev=rand(Normal(0,1),N),
                      forest=rand(Normal(0,1),N));

site_covs[!,:psi] = zeros(N);

mm = ModelMatrix(ModelFrame(@formula(psi~elev+forest), site_covs));

psi_truth = [0, -0.5, 1.2];

psi = logistic.(mm.m * psi_truth);

z = Array{Int}(undef,N); 
for i = 1:N
  z[i] = rand(Binomial(1,psi[i]),1)[1];
end

#Detection

#obs covariates
obs_covs = DataFrame(precip=rand(Normal(0,1),N*J),
                     wind=rand(Normal(0,1),N*J));

obs_covs[!,:p] = zeros(N*J);

mm = ModelMatrix(ModelFrame(@formula(p~precip+wind), obs_covs));

p_truth = [-0.2, 0, 0.7];

p = logistic.(mm.m * p_truth);

y = Array{Int}(undef, N, J);
idx = 1;
for i = 1:N
  for j = 1:J
    y[i,j] = rand(Binomial(1, p[idx] * z[i]))[1];
    global idx += 1;
  end
end

inp = UmData(y, site_covs, obs_covs);

fit = occu(@formula(ψ~elev+forest), @formula(p~precip+wind), inp);

fit

#Truth
truth = round.([psi_truth; p_truth]; digits=3)

yna = Array{Union{Int,Missing}}(y)
yna[1,:] = fill(missing, J)
yna[2,1] = missing

inp2 = UmData(yna, site_covs, obs_covs);

fit2 = occu(@formula(psi~elev+forest), @formula(p~precip+wind), inp2);

fit2

####

nsc = site_covs;

nsc2 = allowmissing(nsc);

nsc2[1,1] = missing;

inp3 = UmData(y, nsc2, obs_covs);

f = @formula(psi~elev+forest)

using DataFrames

m = ModelFrame(f, inp3.site_covs);
m2 = ModelMatrix(m);

##

pr_df = DataFrame(psi=[0,0],elev=[0.5, -0.3], forest=[1,-1])

function predict(fit::Unmarked.UmFit, newdata::DataFrame, param::String) 
  idx = Unmarked.get_index(param, fit.param_names)
  gd = Unmarked.get_design([fit.formulas[idx]], [newdata])
  coefs = fit.coef[fit.inds[idx]]
  lp = gd.mats[1] * coefs
end

pr = predict(fit, pr_df, "psi")


using StatsFuns, ForwardDiff
using Distributions

β = [-0.3, 0.7]
X = transpose([1, -0.4])

#These are the same
trans = logistic.(X*β)
trans2 = map.(x -> cdf(Logistic(), x), X*β)

#solved
function deriv_inv(vals)
  map.(x -> exp(-x) / (1+exp(-x))^2, vals)
end

pred = X * β

gr = [deriv_inv(pred)]

vcov = [[0.3,0.1] [-0.1, 0.5]]

gr * vcov * transpose(gr)

function g(b1, b2)
  b = [b1, b2]
  [logistic.(X*b)]
end

grad = ForwardDiff.jacobian(β -> g(β), β)

f(x,y)=[x^2+y^3-1,x^4 - y^4 + x*y]

a = [1.0,1.0]

ForwardDiff.jacobian(x -> f(x[1],x[2]), a)

function transform_logit(β, X)
  ForwardDiff.jacobian(β -> logistic(X*β), β)
end

transform_logit(β, X)

function logit_mfx(β,x)
  ForwardDiff.jacobian(β-> map(xb -> cdf(Logistic(),xb), x*β), β)  
end

function delta_method(g, θ, Ω)
  dG = ForwardDiff.jacobian(θ->g(θ),θ)
  dG*Ω*dG'  
end

delta_method(β->logit_mfx(β,X)[:,1], β, avar.variance/n)

function j(value::AbstractFloat)
  ForwardDiff.jacobian(x -> logistic(x), value)
end
