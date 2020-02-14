struct RN <: UnmarkedModel
  data::UmData
  opt::UnmarkedOpt
  submodels::NamedTuple
  K::Int
end

"""
    rn(λ_formula::FormulaTerm, p_formula::FormulaTerm, data::UmData, K::Int)

Fit the mixture model of Royle and Nichols (2003), which estimates latent site
abundance using presence-absence data. Probability a species is detected at a
site is assumed to be positively related to species abundance at the site.
Covariates on abundance and detection are specified with `λ_formula` and 
`p_formula`, respectively. The `UmData` object should contain `site_covs` and 
`obs_covs` `DataFrames` containing columns with names matching the formulas. 
The likelihood is marginalized over possible latent abundance values `0:K` for 
each site. `K` should be set high enough that the likelihood of true abundance 
`K` for any site is ≈ 0 (defaults to maximum observed count + 20). Larger `K` 
will increase runtime.
"""
function rn(λ_formula::FormulaTerm, p_formula::FormulaTerm,
            data::UmData, K::Union{Int,Nothing}=nothing)

  abun = UmDesign(:Abundance, λ_formula, LogLink(), data.site_covs)
  det = UmDesign(:Detection, p_formula, LogitLink(), data.obs_covs)
  np = add_idx!([abun, det])

  y = data.y
  N, J = size(y)
  K = isnothing(K) ? maximum(skipmissing(y)) + 20 : K

  function loglik(β::Array)

    λ = transform(abun, β)
    r = transform(det, β)

    ll = zeros(eltype(β), N)

    idx = 0
    for i = 1:N
      fg = zeros(eltype(β),K+1)
      for k = 0:K
        if maximum(skipmissing(y[i,:])) > k
          continue
        end
        fk = pdf(Poisson(λ[i]), k)
        gk = 1.0
        for j = 1:J
          idx += 1
          if ismissing(y[i,j]) continue end
          p = 1 - (1 - r[idx])^k
          gk *= p^y[i,j] * (1-p)^(1-y[i,j])
        end

        fg[k+1] = fk * gk

        #Return r index to start for site if necessary
        if k != K idx -= J end
      end
      ll[i] = log(sum(fg))
    end

    return -sum(ll)
  end

  opt = optimize_loglik(loglik, np)
  RN(data, opt, (abun=UnmarkedSubmodel(abun, opt),
                 det=UnmarkedSubmodel(det, opt)), K)
end
