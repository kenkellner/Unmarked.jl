## Predictions

"""
    predict(um::UnmarkedSubmodel, newx::DataFrame;
            transform::Bool = true, interval::Bool = false, 
            level::Real = 0.95)

Form the predicted response of submodel `um` (e.g., detection or occupancy). 
A DataFrame with new covariate values `newx` can be optionally supplied.

If `transform` is `true`, predicted values are transformed back to the original
scale of the response (e.g., 0-1 for probability of occupancy).
If `interval` is `true`, instead return a tuple of vectors with the prediction 
and the lower and upper confidence bounds for a given `level` 
(0.95 is equivalent to α = 0.05).
"""
function predict(um::UnmarkedSubmodel, newx::DataFrame; transform::Bool=true, 
                 interval::Bool = false, level::Real = 0.95)
  
  dm = UmDesign(um, newx).mat
  est = dm * um.coef

  if !interval
    if !transform return est end
    return invlink(est, um.link)
  end

  vcov = dm * um.vcov * transpose(dm)
  se = sqrt.(diag(vcov))
  zval = -quantile(Normal(), (1-level)/2)
  lower = est - zval * se
  upper = est + zval * se

  if transform  
    est = invlink(est, um.link)
    lower = invlink(lower, um.link)
    upper = invlink(upper, um.link)
  end

  return (prediction=est, lower=lower, upper=upper)
end

function predict(um::UnmarkedSubmodel; transform::Bool=true,
                interval::Bool = false, level::Real = 0.95)
  predict(um, um.data, transform=transform, interval=interval, level=level)
end
