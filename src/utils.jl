"Get number of detections for each site"
function ndetects(y::Array)
  function sum2(row::Array)
    sum(skipmissing(row))
  end
  mapslices(sum2, y, dims=2)
end

"Get index location of provided string in matching array"
function get_index(choice::String, options::Array)
  
  if choice ∉ options
    error(string(choice, " is not a valid option"))
  end

  findall(options .== choice)[1]
end

"Simulate random covariate frame based on provided formula"
function gen_covs(f::FormulaTerm, n::Int)
  covs = collect(map(x -> x.sym, f.rhs))
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

function grad(x::Float64, link::LogitLink)
  exp(-x)/(1+exp(-x))^2
end

function grad(ax::Array, link::LogitLink)
  map(x -> exp(-x)/(1+exp(-x))^2, ax)
end

#Log link
struct LogLink <: Link end

function invlink(x::Float64, link::LogLink)
  exp(x)
end

function invlink(x::Array, link::LogLink)
  exp.(x)
end

function grad(x::Float64, link::LogLink)
  exp(x)
end

function grad(ax::Array, link::LogLink)
  exp.(ax)
end
