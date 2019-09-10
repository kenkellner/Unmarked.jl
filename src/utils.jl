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

## Links

abstract type Link end

#Logit link
struct LogitLink <: Link end

function invlink(x::Float64, link::LogitLink)
  logistic(x)
end

function invlink(x::Array, link::LogitLink)
  logistic.(x)
end

function grad(x::Float64, link::LogitLink)
  exp(-x)/(1+exp(-x))^2
end

function grad(ax::Array, link::LogitLink)
  map(x -> exp(-x)/(1+exp(-x))^2, ax)
end
