module Unmarked

using StatsBase, StatsModels, StatsFuns, Distributions, Random
using LinearAlgebra, DataFrames, Optim, NLSolversBase, ForwardDiff

import Base.show

export UmData, occu, Nmix, predict, simulate

include("utils.jl")
include("data.jl")
include("design.jl")
include("fit.jl")
include("predict.jl")
include("occupancy.jl")
include("nmix.jl")

end
