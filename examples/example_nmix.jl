using Random: seed!, rand
using Distributions: Normal, Poisson
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

site_covs[!,:lam] = zeros(N);

mm = ModelMatrix(ModelFrame(@formula(lam~elev+forest), site_covs));

lam_truth = [0, -0.5, 1.2];

lam = exp.(mm.m * psi_truth);

a = Array{Int}(undef,N); 
for i = 1:N
  a[i] = rand(Poisson(lam[i]),1)[1];
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
    if a[i] == 0;
      y[i,j] = 0;
    else
      y[i,j] = rand(Binomial(a[i], p[idx]))[1];
    end
    global idx += 1;
  end
end

inp = UmData(y, site_covs, obs_covs);


