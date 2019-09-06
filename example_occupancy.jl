using Random: seed!, rand
using Distributions: Normal, Binomial
using DataFrames: DataFrame, nrow
using StatsModels: @formula, ModelFrame, ModelMatrix
using StatsFuns: logistic, logit

include("design.jl")
include("fit.jl")
include("occupancy.jl")

seed!(123);

#Simulate data

#state covariates
elev = rand(Normal(0,1),1000);
forest = rand(Normal(0,1),1000);

site_covs = DataFrame(elev=rand(Normal(0,1),1000),
                      forest=rand(Normal(0,1),1000));

site_covs[!,:psi] = zeros(nrow(site_covs));

f = @formula(psi~elev+forest+fake)

mm = ModelMatrix(ModelFrame(@formula(psi~elev+forest), site_covs));

psi_truth = [0, -0.5, 1.2];

psi = logistic.(mm.m * psi_truth);

z = Array{Int32}(undef,1000); 
for i = 1:1000
  z[i] = rand(Binomial(1,psi[i]),1)[1];
end

p = 0.4;
J = 5;
y = Array{Int32}(undef, 1000);
for i = 1:length(y)
  y[i] = rand(Binomial(J, p * z[i]));
end

inp = umf(y, site_covs);

fit = fit_model(@formula(psi~elev+forest), inp);

fit

#Truth
truth = round.([psi_truth; logit(p)]; digits=3)
