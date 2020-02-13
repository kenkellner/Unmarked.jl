struct Nmix <: UnmarkedModel
  data::UmData
  opt::UnmarkedOpt
  submodels::NamedTuple
  K::Int
end

"""
    nmix(λ_formula::FormulaTerm, p_formula::FormulaTerm, data::UmData, K::Int)

Fit single-season N-mixture models. Covariates on abundance and detection 
are specified with `λ_formula` and `p_formula`, respectively. The `UmData`
object should contain `site_covs` and `obs_covs` `DataFrames` containing
columns with names matching the formulas. The likelihood is marginalized over
possible latent abundance values `0:K` for each site. `K` should be set high
enough that the likelihood of true abundance `K` for any site is ≈ 0 (defaults 
to maximum observed count + 20). Larger `K` will increase runtime.
"""
function nmix(λ_formula::FormulaTerm, p_formula::FormulaTerm, 
              data::UmData, K::Union{Int,Nothing}=nothing)
  
  abun = UmDesign(:Abundance, λ_formula, LogLink(), data.site_covs)
  det = UmDesign(:Detection, p_formula, LogitLink(), data.obs_covs)
  np = add_idx!([abun, det])

  y = data.y
  N, J = size(y)
  K = isnothing(K) ? maximum(skipmissing(y)) + 20 : K

  function loglik(β::Array)
  
    λ = transform(abun, β)
    p = transform(det, β)
   
    ll = zeros(eltype(β),N)
    
    idx = 0
    for i = 1:N

      if all(ismissing.(y[i,:]))
        idx += J
        continue
      end
      
      fg = zeros(eltype(β),K+1)
      for k = 0:K       
        if maximum(skipmissing(y[i,:])) > k
          continue
        end
        fk = pdf(Poisson(λ[i]), k)
        gk = 1.0
        for j = 1:J
          idx += 1
          if ismissing(y[i,j])
            continue
          end
          gk *= pdf(Binomial(k, p[idx]), y[i,j])        
        end
        fg[k+1] = fk * gk
        
        #Return p index to start for site if necessary
        if k != K idx -= J end
      end
      ll[i] = log(sum(fg))
    end

    return -sum(ll)
  end

  opt = optimize_loglik(loglik, np)
  Nmix(data, opt, (abun=UnmarkedSubmodel(abun,opt), 
            det=UnmarkedSubmodel(det,opt)), K)
end

#Fit multiple models at once
"""
    nmix(λ_formula::Union{Array,FormulaTerm}, 
         p_formula::Union{Array,FormulaTerm}, data::UmData, K::Int)

Provide multiple formulas for λ and/or p and fit models for all formula
combinations. To fit all model subsets, wrap formula with `allsubs()`.
"""

function nmix(λ_form::Union{Array,FormulaTerm}, 
              p_form::Union{Array,FormulaTerm}, 
              data::UmData, K::Union{Int,Nothing}=nothing)
  cf = combine_formulas(λ_form, p_form)
  out = Array{UnmarkedModel}(undef, length(cf))
  p = Progress(length(cf), desc=string("Fitting ", length(cf), " models "))
  Threads.@threads for i in 1:length(cf) 
    out[i] = nmix(cf[i][1],cf[i][2], data, K)
    next!(p)
  end
  return UnmarkedModels(out)
end

# Simulations------------------------------------------------------------------

"Simulate new dataset from a fitted model"
function simulate(fit::Nmix)
  
  ydims = collect(size(fit.data.y))
  λ = predict(abundance(fit))
  p = predict(detection(fit))
  
  y = sim_y(Nmix, λ, p)
  rep_missing!(y, fit.data.y)
 
  UmData(y, fit.data.site_covs, fit.data.obs_covs)
end

"Simulate new dataset from provided formulas"
function simulate(::Type{Nmix}, λ_form::FormulaTerm, p_form::FormulaTerm, 
                  ydims::Tuple{Int,Int}, coef::Array)
 
  sc = gen_covs(λ_form, ydims[1])
  oc = gen_covs(p_form, prod(ydims))

  abun = UmDesign(:Abun, λ_form, LogLink(), sc)
  det = UmDesign(:Det, p_form, LogitLink(), oc)
  np = add_idx!([abun, det])

  if np != length(coef) error(string("Coef array must be length ",np)) end

  λ = transform(abun, coef)
  p = transform(det, coef)

  y = sim_y(Nmix, λ, p)

  UmData(y, sc, oc)
end

function sim_y(::Type{Nmix}, λ::Array, p::Array)
  
  N = length(λ)
  J = Integer(length(p) / N)
  z = map(x -> rand(Poisson(x))[1], λ)
  y = Array{Union{Int,Missing}}(undef, N, J)
  idx = 0
  for i = 1:N
    for j = 1:J
      idx += 1
      y[i,j] = z[i] == 0 ? 0 : rand(Binomial(z[i], p[idx]))[1]
    end
  end

  return y
end

# Update ----------------------------------------------------------------------

function update(fit::Nmix, data::UmData)
  nmix(abundance(fit).formula, detection(fit).formula, data, fit.K)
end

# Goodness-of-fit -------------------------------------------------------------

function fitted(fit::Nmix)
  y = fit.data.y
  N,J = size(fit.data.y)
  λ = repeat(predict(abundance(fit)), inner=J)
  return λ .* predict(detection(fit))
end

function residuals(fit::Nmix)
  ft = fitted(fit)
  yl = reshape(transpose(fit.data.y), length(fit.data.y))
  return ft .- yl
end

function chisq(fit::Nmix)
  return sum(skipmissing(residuals(fit) .^ 2 ./ fitted(fit)))
end

"""
    gof(fit::Nmix, nsims::Int=30)

Compute an N-mixture goodness-of-fit test using Pearson residuals.
The distribution of the test statistic is generated using parametric 
bootstrapping with `nsims` simulations.
"""
function gof(fit::Nmix, nsims::Int=30)
  test_name = "N-mixture Goodness-of-fit"
  t0 = chisq(fit)
  tstar = parboot(fit, nsims, chisq)
  return UmGOF(test_name,t0, tstar)
end
