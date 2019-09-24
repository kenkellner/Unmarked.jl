module Unmarked

using StatsBase, StatsModels, StatsFuns, Distributions, Random, Printf
using LinearAlgebra, DataFrames, Optim, NLSolversBase, ForwardDiff
using Combinatorics: combinations
using PrettyTables: pretty_table, ft_printf
using ProgressMeter: @showprogress

import Base: show, getindex, length
import StatsBase: aic, aicc, bic, coef, coefnames, coeftable, deviance, dof,
                  loglikelihood, modelmatrix, predict, nobs, stderror, vcov

export UmData, Occu, Nmix
export @formula
export aic, aicc, bic, coef, coefnames, coeftable, deviance, dof, gof, 
       loglikelihood, modelmatrix, nobs, predict, simulate, stderror, vcov

#Fitting functions       
export occu, nmix

#Submodel extractors
export detection, occupancy, abundance

#Utility functions
export allsub

include("utils.jl")
include("links.jl")
include("formula.jl")
include("data.jl")
include("design.jl")
include("fit.jl")
include("gof.jl")
include("predict.jl")
include("occupancy.jl")
include("nmix.jl")

end
