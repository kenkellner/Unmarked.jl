# Unmarked.jl

[![Build Status](https://travis-ci.org/kenkellner/Unmarked.jl.svg?branch=master)](https://travis-ci.org/kenkellner/Unmarked.jl)
[![codecov](https://codecov.io/gh/kenkellner/Unmarked.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/kenkellner/Unmarked.jl)

Unofficial Julia port of the R package [unmarked](https://cran.r-project.org/web/packages/unmarked/index.html), for analyzing ecological data while accounting for imperfect detection.

```julia
using Unmarked, DataFrames

#Simulate an occupancy dataset
ψ_formula = @formula(ψ~elev+forest);
p_formula = @formula(p~precip+wind);
β_truth = [0, -0.5, 1.2, -0.2, 0, 0.7];
umd = simulate(Occu, ψ_formula, p_formula, [1000, 5], β_truth);

#Fit the model
fit = occu(ψ_formula, p_formula, umd)

#Compare with true coefficient values
hcat(coef(fit), β_truth)

#Predict occupancy probabilities from DataFrame
newdata = DataFrame(elev=[0.5, -0.3], forest=[1,-1]);
predict(occupancy(fit), newdata)

#Goodness-of-fit
gof(fit)

#Fit all subsets of model covariates and compare with AIC
occu(allsub(ψ_formula), allsub(p_formula), umd)
```
