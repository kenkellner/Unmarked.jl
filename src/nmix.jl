struct Nmix <: UnmarkedModel
  data::UmData
  opt::UnmarkedOpt
  submodels::NamedTuple
end

"Fit N-mixture models"
function nmix(λ_formula::FormulaTerm, p_formula::FormulaTerm, 
              data::UmData, K::Int)
  
  abun = UmDesign(:Abundance, λ_formula, LogLink(), data.site_covs)
  det = UmDesign(:Detection, p_formula, LogitLink(), data.obs_covs)
  add_idx!([abun, det])
  np = det.idx.stop

  y = data.y
  N, J = size(y)

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
        if maximum(y[i,:]) > k
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
  UmFitNmix(data, opt, (abun=UnmarkedSubmodel(abun,opt), 
                  det=UnmarkedSubmodel(det,opt)))

end

function nmix(λ_formula::FormulaTerm, p_formula::FormulaTerm, 
              data::UmData)

  K = maximum(data.y) + 20
  nmix(λ_formula, p_formula, data, K)
  
end

