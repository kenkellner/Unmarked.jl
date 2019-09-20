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
