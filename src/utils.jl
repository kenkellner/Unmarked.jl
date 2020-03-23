#Get number of detections for each site
function ndetects(y::Array)
  function sum2(row::Array)
    sum(skipmissing(row))
  end
  mapslices(sum2, y, dims=2)
end

#Simulate random covariate frame based on provided formula
function gen_covs(f::FormulaTerm, n::Int)  
  covs = varnames(f).rhs
  if isnothing(covs) return DataFrame(_dummy=ones(n)) end 
  nc = length(covs)
  out = DataFrame(reshape(rand(Normal(0,1), n*nc), n, nc))
  DataFrames.rename!(out, covs)
  out
end

#Replicate missing values in a new simulated y matrix
function rep_missing!(ynew::Array, y::Array)
  na_idx = findall(ismissing.(y))
  if length(na_idx) == 0 return nothing end
  ynew[na_idx] = fill(missing, length(na_idx))
end

#Get appropriate value of K
function check_K(K::Int, y::Array{Union{Missing,Int},2})
  if(K < (maximum(skipmissing(y))))
    error("K should be larger than the maximum observed abundance")
  end
  K
end

function check_K(K::Nothing, y::Array{Union{Missing,Int},2})
  maximum(skipmissing(y)) + 20
end

#Get minimum bound for integrating over possible abundance values K
#Minimum bound = maximum abundance ever observed at a site
function get_Kmin(y::Array{Union{Missing,Int64},2})
  N = size(y, 1)
  kmin = zeros(Int64, N)
  for n in 1:N
    if(all(ismissing.(y[n, :]))) continue end 
    kmin[n] = maximum(skipmissing(y[n, :]))
  end
  kmin
end
