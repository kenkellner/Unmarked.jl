"Fit single-season occupancy models"
function occu(ψ_formula::FormulaTerm, p_formula::FormulaTerm, data::UmData)

  mods = ["Occupancy", "Detection"]
  formulas = [ψ_formula, p_formula]
  links = [LogitLink(), LogitLink()]

  y = data.y
  N = size(y)[1]
  J = size(y)[2]
  nd = ndetects(y)

  gd = get_design(formulas, [data.site_covs, data.obs_covs])
  
  np = length(gd.coefs)

  function loglik(β::Array)
    
    ψ = logistic.(gd.mats[1] * β[gd.inds[1]])
    p = logistic.(gd.mats[2] * β[gd.inds[2]])

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

  UmFit(opt.coef, opt.se, opt.vcov, opt.AIC,
        gd.coefs, mods, gd.params, gd.inds, formulas, links)

end
