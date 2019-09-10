module Unmarked

using StatsBase, StatsModels, StatsFuns, Distributions
using LinearAlgebra, DataFrames, Optim, NLSolversBase, ForwardDiff

import Base.show

export UmData, occu, predict

include("utils.jl")
include("data.jl")
include("design.jl")
include("fit.jl")
include("predict.jl")
include("occupancy.jl")

end
