struct Occu <: UnmarkedModel
  data::UmData
  opt::UnmarkedOpt
  submodels::NamedTuple
end

"""
    occu(ψ_form::FormulaTerm, p_form::FormulaTerm, data::Umdata)

Fit single-season occupancy models. Covariates on occupancy and detection 
are specified with `ψ_form` and `p_form`, respectively. The `UmData`
object should contain `site_covs` and `obs_covs` `DataFrames` containing
columns with names matching the formulas.
"""
function occu(ψ_form::FormulaTerm, p_form::FormulaTerm, data::UmData)
  
  occ = UmDesign(:Occupancy, ψ_form, LogitLink(), data.site_covs)
  det = UmDesign(:Detection, p_form, LogitLink(), data.obs_covs)
  np = add_idx!([occ, det])

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

"""
    occu(ψ_formula::Union{Array,FormulaTerm}, 
         p_formula::Union{Array,FormulaTerm}, data::UmData)

Provide multiple formulas for ψ and/or p and fit models for all formula
combinations. To fit all model subsets, wrap formula with `allsubs()`.
"""
function occu(ψ_form::Union{Array,FormulaTerm}, 
              p_form::Union{Array,FormulaTerm}, data::UmData)
  cf = combine_formulas(ψ_form, p_form)
  msg = string("Fitting ", length(cf), " models ")
  out = Array{UnmarkedModel}(undef, length(cf))
  @showprogress 1 msg for i in 1:length(cf)
    out[i] = occu(cf[i][1], cf[i][2], data)
  end
  return UnmarkedModels(out)
end

# Simulations------------------------------------------------------------------

"Simulate new dataset from a fitted model"
function simulate(fit::Occu)
  
  ψ = predict(occupancy(fit))
  p = predict(detection(fit))
 
  y = sim_y(Occu, ψ, p)
  rep_missing!(y, fit.data.y)
  
  UmData(y, fit.data.site_covs, fit.data.obs_covs)
end

"Simulate new dataset from provided formulas"
function simulate(::Type{Occu}, ψ_form::FormulaTerm, p_form::FormulaTerm, 
                  ydims::Tuple{Int,Int}, coef::Array)
 
  sc = gen_covs(ψ_form, ydims[1])
  oc = gen_covs(p_form, prod(ydims))

  occ = UmDesign(:Occ, ψ_form, LogitLink(), sc)
  det = UmDesign(:Det, p_form, LogitLink(), oc)
  np = add_idx!([occ, det])

  if np != length(coef) error(string("Coef array must be length ",np)) end

  ψ = transform(occ, coef)
  p = transform(det, coef)

  y = sim_y(Occu, ψ, p)

  UmData(y, sc, oc)
end

#Simulate y matrix from provided psi and p vectors"
function sim_y(::Type{Occu}, ψ::Array, p::Array)
  
  N = length(ψ)
  J = Integer(length(p) / N)
  z = map(x -> rand(Binomial(1, x))[1], ψ)
  y = Array{Union{Int,Missing}}(undef, N, J)
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

#Get cohorts with unique NA placement
function get_cohorts(fit::Occu)
  data = fit.data
  na_pattern = mapslices(x -> string(ismissing.(x)*1...), data.y, dims=2)[:,1]
  cohorts = string.(keys(countmap(na_pattern)))

  out = []
  for i in 1:length(cohorts)
    #Discard cohort where all obs are NAs
    if cohorts[i] == repeat("1", size(data.y)[2]) continue end
    idx = findall(na_pattern .== cohorts[i])
    push!(out, data[idx])
  end

  return out
end
  
#Get probability of a particular encounter history
function get_prob_eh(::Type{Occu}, eh::String, pvec::Array, psi::Float64)
  ehv = map(x -> x == "missing" ? missing : parse(Int, x), split(eh, " "))
  pvals = (ehv .* pvec) + (1 .- ehv) .* (1 .- pvec)
  pvals = collect(skipmissing(pvals))
  
  if sum(skipmissing(ehv)) == 0
    return psi * prod(pvals) + (1-psi)
  end
  
  return psi * prod(pvals)
end

#Get counts for each observed encounter history
function get_obs_counts(fit::Occu, data::UmData)
  eh_strings = mapslices(x -> join(x, " "), data.y, dims=2)
  return countmap(eh_strings)
end

#Get expected counts of each observed encounter history
function get_exp_counts(fit::Occu, data::UmData)
  N,J = size(data.y)

  obs_counts = get_obs_counts(fit, data)

  eh_obs = collect(keys(obs_counts))
  nobs = length(eh_obs)

  psi = predict(occupancy(fit), data.site_covs)
  p = predict(detection(fit), data.obs_covs)

  counts_expect = fill(0.0, nobs)
  for i = 1:nobs
    pstart = 1
    
    for j = 1:N
      pend = pstart + J - 1
      counts_expect[i] += get_prob_eh(Occu, eh_obs[i], p[pstart:pend], psi[j])
      pstart += J
    end
  end

  return Dict(zip(eh_obs, counts_expect))
end

#Compute MacKenzie-Bailey (2004) chi-square statistic
function mb_chisq(fit::Occu)
  
  #Divide dataset into cohorts based on NA placement
  cohorts = get_cohorts(fit::Occu)

  chisq_all = fill(0.0, length(cohorts))
  for i in 1:length(cohorts)

    N = size(cohorts[i].y)[1]

    obs_dict = get_obs_counts(fit, cohorts[i])
    exp_dict = get_exp_counts(fit, cohorts[i])

    neh = length(obs_dict)

    obs = collect(values(obs_dict))
    exp = collect(values(exp_dict))

    #For observed encounter histories
    chisq_prt1 = sum(map(x -> (obs[x] - exp[x])^2 / exp[x], 1:neh))

    #For unobserved possible encounter histories
    chisq_prt2 = maximum([0, N - sum(exp)])

    chisq_all[i] = chisq_prt1 + chisq_prt2
  end

  return sum(chisq_all)
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
