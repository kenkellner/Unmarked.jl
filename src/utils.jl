#Extract variable names from a formula before creating schema
function varnames(f::FormulaTerm)
  rhs = varnames(f.rhs)
  if rhs isa Symbol rhs = [rhs] end
  return (lhs=varnames(f.lhs), rhs=rhs)
end

function varnames(f::Term)
  return Symbol(f)
end

function varnames(f::StatsModels.TupleTerm)
  raw = map(x -> varnames(x), f)
  return get_unique(raw)
end

function varnames(f::ConstantTerm)
  return nothing
end

function varnames(f::InteractionTerm)
  return map(x -> varnames(x), f.terms) 
end

function get_unique(f::Tuple)
  out = Symbol[]
  fl(v) = for x in v
    if x isa Tuple fl(x) else push!(out, x) end
  end
  fl(f)
  return unique(out)
end

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

## Links

abstract type Link end

#Logit link
struct LogitLink <: Link end

function invlink(x::Float64, link::LogitLink)
  logistic(x)
end

function invlink(ax::Array, link::LogitLink)
  logistic.(ax)
end

#Log link
struct LogLink <: Link end

function invlink(x::Float64, link::LogLink)
  exp(x)
end

function invlink(x::Array, link::LogLink)
  exp.(x)
end
