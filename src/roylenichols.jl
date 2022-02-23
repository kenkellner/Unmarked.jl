struct RN <: UnmarkedModel
  data::UmData
  opt::UnmarkedOpt
  submodels::NamedTuple
  K::Int
end

"""
    rn(λ_formula::FormulaTerm, p_formula::FormulaTerm, data::UmData; K::Int)

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
            data::UmData; K::Union{Int,Nothing}=nothing)

  abun = UmDesign(:Abundance, λ_formula, LogLink(), data.site_covs)
  det = UmDesign(:Detection, p_formula, LogitLink(), data.obs_covs)
  add_idx!([abun, det])
  np = get_np([abun, det])

  y = data.y
  N, J = size(y)
  K = check_K(K, y)
  Kmin = get_Kmin(y)

  function loglik(β::Array)

    λ = transform(abun, β)
    r = transform(det, β)

    ll = zeros(eltype(β), N)

    Threads.@threads for i = 1:N
      rsub = r[(i*J-J+1):(i*J)]
      fg = zeros(eltype(β),K+1)
      for k = Kmin[i]:K
        fk = pdf(Poisson(λ[i]), k)
        gk = 1.0
        for j = 1:J
          if ismissing(y[i,j]) continue end
          p = 1 - (1 - rsub[j])^k
          gk *= p^y[i,j] * (1-p)^(1-y[i,j])
        end
        fg[k+1] = fk * gk
      end
      ll[i] = log(sum(fg))
    end

    return -sum(ll)
  end

  opt = optimize_loglik(loglik, np)
  RN(data, opt, (abun=UnmarkedSubmodel(abun, opt),
                 det=UnmarkedSubmodel(det, opt)), K)
end
