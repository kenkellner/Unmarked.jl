using Unmarked, Random, DataFrames

Random.seed!(123);

ψ_formula = @formula(ψ~elev+forest);
p_formula = @formula(p~precip+wind);
β_truth = [0, -0.5, 1.2, -0.2, 0, 0.7];

umd = simulate(Occu, ψ_formula, p_formula, [1000, 5], β_truth);

fit = occu(ψ_formula, p_formula, umd);

hcat(coef(fit), β_truth)

#Prediction
pr = predict(detection(fit));

pr_df = DataFrame(elev=[0.5, -0.3], forest=[1,-1]);

predict(occupancy(fit), pr_df, interval=true) 

#Goodness-of-fit
gof(fit)

#Missing values
yna = Array{Union{Int,Missing}}(y)
yna[1,:] = fill(missing, J)
yna[2,1] = missing

inp2 = UmData(yna, site_covs, obs_covs);

fit2 = occu(@formula(psi~elev+forest), @formula(p~precip+wind), inp2)
