using Unmarked, DataFrames

λ_formula = @formula(λ~elev+forest);
p_formula = @formula(p~precip+wind);
β_truth = [0, -0.3, 0.7, -0.2, 0, 0.7];

umd = simulate(Nmix, λ_formula, p_formula, (1000, 5), β_truth);

fit = nmix(λ_formula, p_formula, umd);

hcat(coef(fit), β_truth)

#Prediction
pr_df = DataFrame(elev=[0.5, -0.3], forest=[1,-1]);

predict(abundance(fit), pr_df, interval=true) 

#Goodness-of-fit
gof(fit)

#Fit all subsets of covariates
fit_all = nmix(allsub(λ_formula), allsub(p_formula), umd);

#Model selection table
fit_all

#Missing values
yna = Array{Union{Int,Missing}}(deepcopy(umd.y))
yna[1,:] = fill(missing, 5)
yna[2,1] = missing

inp2 = UmData(yna, umd.site_covs, umd.obs_covs);

fit2 = nmix(@formula(psi~elev+forest), @formula(p~precip+wind), inp2)

gof(fit2)
