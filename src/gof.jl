#Goodness-of-fit output structure
struct UmGOF
  test_name::String
  t0::Float64
  tstar::Array{Float64}
end

#Parametric bootstrap
function parboot(fit::UnmarkedModel, nsims::Int, statistic::Any)

  out = Array{Float64}(undef, nsims)

  @showprogress 1 string("Bootstrap (",nsims," sims) ") for i = 1:nsims
    new_data = simulate(fit)
    new_fit = update(fit, new_data)
    out[i] = statistic(new_fit)
  end

  return out
end
