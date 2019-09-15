struct Occu <: UnmarkedModel
  data::UmData
  opt::UnmarkedOpt
  submodels::NamedTuple
end

"""
    occu(ψ_formula::FormulaTerm, p_formula::FormulaTerm, data::Umdata)

Fit single-season occupancy models. Covariates on occupancy and detection 
are specified with `ψ_formula` and `p_formula`, respectively. The `UmData`
object should contain `site_covs` and `obs_covs` `DataFrames` containing
columns with names matching the formulas.
"""
function occu(ψ_formula::FormulaTerm, p_formula::FormulaTerm, data::UmData)
  
  occ = UmDesign(:Occupancy, ψ_formula, LogitLink(), data.site_covs)
  det = UmDesign(:Detection, p_formula, LogitLink(), data.obs_covs)
  add_idx!([occ, det])
  np = det.idx.stop

  y = data.y
  N, J = size(y)
  nd = ndetects(y)

  function loglik(β::Array)
    
    ψ = transform(occ, β)
    p = transform(det, β)

    ll = zeros(eltype(β), N)

    idx = 0
    for i = 1:N
      if all(ismissing.(y[i,:]))
        idx += J
        continue
      end
      cp = 1.0
      for j = 1:J
        idx += 1
        if ismissing(y[i,j])
          continue
        end
        cp *= p[idx]^y[i,j] * (1-p[idx])^(1-y[i,j])
      end
      ll[i] = log(ψ[i] * cp + (1-ψ[i]) * (nd[i]==0))
    end

    return -sum(ll)
  end

  opt = optimize_loglik(loglik, np)
  Occu(data, opt, (occ=UnmarkedSubmodel(occ,opt),
                   det=UnmarkedSubmodel(det,opt)))
end

#Fit alias
"""
    fit(Occu, ψ_formula::FormulaTerm, p_formula::FormulaTerm, data::Umdata)

Fit single-season occupancy models. Covariates on occupancy and detection 
are specified with `ψ_formula` and `p_formula`, respectively. The `UmData`
object should contain `site_covs` and `obs_covs` `DataFrames` containing
columns with names matching the formulas.
"""
function fit(::Type{Occu}, ψ_formula::FormulaTerm, p_formula::FormulaTerm,
             data::UmData)
  occu(ψ_formula, p_formula, data)
end

# Simulations------------------------------------------------------------------

"Simulate new dataset from a fitted model"
function simulate(fit::Occu)
  
  ydims = collect(size(fit.data.y))
  ψ = predict(occupancy(fit))
  p = predict(detection(fit))
  
  y = _sim_y_occu(ψ, p, ydims)
  
  UmData(y, fit.data.site_covs, fit.data.obs_covs)
end

"Simulate new dataset from provided formulas"
function simulate(::Type{Occu}, ψ_formula::FormulaTerm, 
                  p_formula::FormulaTerm, ydims::Array{Int}, coef::Array)
 
  sc = gen_covs(ψ_formula, ydims[1])
  oc = gen_covs(p_formula, ydims[1] * ydims[2])

  occ = UmDesign(:Occ, ψ_formula, LogitLink(), sc)
  det = UmDesign(:Det, p_formula, LogitLink(), oc)
  add_idx!([occ, det])

  np = det.idx.stop
  if np != length(coef)
    error(string("Coef array must be length ",np))
  end

  ψ = transform(occ, coef)
  p = transform(det, coef)

  y = _sim_y_occu(ψ, p, ydims)

  UmData(y, sc, oc)
end

#Simulate y matrix from provided psi and p vectors"
function _sim_y_occu(ψ::Array, p::Array, ydims::Array)
  
  N,J = ydims
  z = map(x -> rand(Binomial(1, x))[1], ψ)
  y = Array{Int}(undef, N, J)
  idx = 0
  for i = 1:N
    for j = 1:J
      idx += 1
      y[i,j] = rand(Binomial(1, p[idx] * z[i]))[1]
    end
  end

  return y
end

# Update ----------------------------------------------------------------------

function update(fit::Occu, data::UmData)
  occu(occupancy(fit).formula, detection(fit).formula, data)
end

# Goodness-of-fit -------------------------------------------------------------

#Get probability of a particular encounter history
function get_prob_eh(eh::String, pvec::Array, psi::Float64)
  ehv =  parse.(Int, split(eh,""))
  pvals = (ehv .* pvec) + (1 .- ehv) .* (1 .- pvec)
  
  if sum(ehv) == 0
    return psi * prod(pvals) + (1-psi)
  end
  
  return psi * prod(pvals)
end

#Get counts for each observed encounter history
function get_obs_counts(fit::Occu)
  eh_strings = mapslices(x -> string(x...), fit.data.y, dims=2)
  return countmap(eh_strings)
end

#Get expected counts of each observed encounter history
function get_exp_counts(fit::Occu)
  N,J = size(fit.data.y)

  obs_counts = get_obs_counts(fit)

  eh_obs = collect(keys(obs_counts))
  nobs = length(eh_obs)

  psi = predict(occupancy(fit))
  p = predict(detection(fit))

  counts_expect = fill(0.0, nobs)
  for i = 1:nobs
    pstart = 1
    
    for j = 1:N
      pend = pstart + J - 1
      counts_expect[i] += get_prob_eh(eh_obs[i], p[pstart:pend], psi[j])
      pstart += J
    end
  end

  return Dict(zip(eh_obs, counts_expect))
end

#Compute MacKenzie-Bailey (2004) chi-square statistic
function mb_chisq(fit::Occu)
  
  N = size(fit.data.y)[1]

  obs_dict = get_obs_counts(fit)
  exp_dict = get_exp_counts(fit)

  neh = length(obs_dict)

  obs = collect(values(obs_dict))
  exp = collect(values(exp_dict))

  #For observed encounter histories
  chisq_prt1 = sum(map(x -> (obs[x] - exp[x])^2 / exp[x], 1:neh))

  #For unobserved encounter histories
  chisq_prt2 = maximum([0, N - sum(exp)])

  return chisq_prt1 + chisq_prt2
end

"""
    gof(fit::Occu, nsims::Int=50)

Compute the MacKenzie-Bailey (2004) goodness-of-fit test for a
single-season occupancy model. The distribution of the test statistic
is generated using parametric bootstrapping with `nsims` simulations.
"""
function gof(fit::Occu, nsims::Int=50)
  test_name = "MacKenzie-Bailey Goodness-of-fit"
  t0 = mb_chisq(fit)
  tstar = parboot(fit, nsims, mb_chisq)
  return UmGOF(test_name,t0, tstar)
end

function Base.show(io::IO, gof::UmGOF)
  pval = mean(gof.tstar .> gof.t0)
  chat = gof.t0 / mean(gof.tstar)
  println()
  println(gof.test_name)
  println()
  println(string("  χ2 = ", @sprintf("%.4f",round(gof.t0, digits=4))))
  println(string("  P-value = ", @sprintf("%.4f",round(pval, digits=4))))
  print(string("  Est. c-hat = ", 
               @sprintf("%.4f", round(chat, digits=4))))
end
