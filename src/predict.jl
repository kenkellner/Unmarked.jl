## Predictions

"Structure to hold predicted values"
struct UmPred
  name::Symbol
  Predicted::Array
  SE::Array
  lower::Array
  upper::Array
  transform::Bool
end

"Predict values, optionally using a new covariate data frame"
function predict(um::UmModel, newdata::DataFrame, 
                 transform::Bool=true)
  dm = UmDesign(um.name, um.formula, um.link, newdata).mat
  est = dm * um.coef
  vcov = dm * um.vcov * transpose(dm)
  se = sqrt.(diag(vcov))
  lower = est - 1.96 * se
  upper = est + 1.96 * se
  if transform
    est = invlink(est, um.link)
    g = grad(est, um.link)
    vcov = Diagonal(g) * vcov * Diagonal(g)
    se = sqrt.(diag(vcov))
    lower = Unmarked.invlink(lower, um.link)
    upper = Unmarked.invlink(upper, um.link)
  end

  UmPred(um.name, est, se, lower, upper, transform)
end

function predict(um::UmModel, transform::Bool=true)
  predict(um, um.data, transform)
end

function Base.show(io::IO, up::UmPred)
  println(DataFrame(Predicted=up.Predicted, SE=up.SE, 
                    lower=up.lower, upper=up.upper))
end
