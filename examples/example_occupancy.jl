using Unmarked, Random, StatsModels, DataFrames

Random.seed!(123);

ψ_formula = @formula(ψ~elev+forest);
p_formula = @formula(p~precip+wind);
β_truth = [0, -0.5, 1.2, -0.2, 0, 0.7];

umd = simulate(Unmarked.UmSimOccu(), ψ_formula, p_formula, 
               [1000, 5], β_truth);

fit = occu(ψ_formula, p_formula, umd);

fit

#Prediction
pr = predict(fit.models.det);

pr_df = DataFrame(elev=[0.5, -0.3], forest=[1,-1]);

predict(fit.models.occ, pr_df) 

#Simulate
s = simulate(fit);

#Missing values
yna = Array{Union{Int,Missing}}(y)
yna[1,:] = fill(missing, J)
yna[2,1] = missing

inp2 = UmData(yna, site_covs, obs_covs);

fit2 = occu(@formula(psi~elev+forest), @formula(p~precip+wind), inp2);

fit2


