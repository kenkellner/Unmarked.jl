using Unmarked, DataFrames

ψ_formula = @formula(ψ~elev+forest);
p_formula = @formula(p~precip+wind);
β_truth = [0, -0.5, 1.2, -0.2, 0, 0.7];

umd = simulate(Occu, ψ_formula, p_formula, (1000, 5), β_truth);

fit = occu(ψ_formula, p_formula, umd);

hcat(coef(fit), β_truth)

#Plots
using Gadfly
set_default_plot_size(20cm, 20cm)
whiskerplot(fit)
effectsplot(fit)

#Prediction
pr = predict(detection(fit));

pr_df = DataFrame(elev=[0.5, -0.3], forest=[1,-1]);

predict(occupancy(fit), pr_df, interval=true) 

#Goodness-of-fit
gof(fit)

#Fit all subsets of covariates
fit_all = occu(allsub(ψ_formula), allsub(p_formula), umd);

#Model selection table
fit_all

#Missing values
yna = Array{Union{Int,Missing}}(deepcopy(umd.y))
yna[1,:] = fill(missing, 5)
yna[2,1] = missing

inp2 = UmData(yna, umd.site_covs, umd.obs_covs);

fit2 = occu(@formula(psi~elev+forest), @formula(p~precip+wind), inp2)

gof(fit2)
