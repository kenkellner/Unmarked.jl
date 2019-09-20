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
