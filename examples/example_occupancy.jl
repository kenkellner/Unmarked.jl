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

fit = occu(@formula(ψ~elev*forest), @formula(p~precip+wind), inp);

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
  gd.mats[1] * coefs
end

pr = predict(fit, pr_df, "psi")

