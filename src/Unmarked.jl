module Unmarked

using StatsBase, StatsModels, StatsFuns, Distributions, Random
using LinearAlgebra, DataFrames, Optim, NLSolversBase, ForwardDiff

import Base.show
import StatsBase: aic, aicc, bic, coef, coefnames, coeftable, deviance, dof,
                  loglikelihood, nobs, stderror, vcov

export UmData, Occu, Nmix
export @formula
export aic, aicc, bic, coef, coefnames, coeftable, deviance, dof, 
       loglikelihood, nobs, predict, simulate, stderror, vcov

#Fitting functions       
export occu, nmix

#Submodel extractors
export detection, occupancy, abundance

include("utils.jl")
include("data.jl")
include("design.jl")
include("fit.jl")
include("predict.jl")
include("occupancy.jl")
include("nmix.jl")

end
