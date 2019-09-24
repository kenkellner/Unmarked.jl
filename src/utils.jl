#Get number of detections for each site
function ndetects(y::Array)
  function sum2(row::Array)
    sum(skipmissing(row))
  end
  mapslices(sum2, y, dims=2)
end

#Simulate random covariate frame based on provided formula"
function gen_covs(f::FormulaTerm, n::Int)  
  covs = varnames(f).rhs
  if isnothing(covs) return DataFrame(_dummy=ones(n)) end 
  nc = length(covs)
  out = DataFrame(reshape(rand(Normal(0,1), n*nc), n, nc))
  DataFrames.names!(out, covs)
  out
end

#Replicate missing values in a new simulated y matrix
function rep_missing!(ynew::Array, y::Array)
  na_idx = findall(ismissing.(y))
  if length(na_idx) == 0 return nothing end
  ynew[na_idx] = fill(missing, length(na_idx))
end
