#Goodness-of-fit output structure
struct UmGOF
  test_name::String
  t0::Float64
  tstar::Array{Float64}
end

#Parametric bootstrap
function parboot(fit::UnmarkedModel, nsims::Int, statistic::Any)

  out = Array{Float64}(undef, nsims)
  p = Progress(nsims, desc=string("Bootstrap (",nsims," sims) "))
  Threads.@threads for i = 1:nsims
    new_data = simulate(fit)
    new_fit = update(fit, new_data)
    out[i] = statistic(new_fit)
    next!(p)
  end

  return out
end
