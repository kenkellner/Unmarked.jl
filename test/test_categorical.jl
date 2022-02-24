@testset "Cat Covs" begin

  using CategoricalArrays

  #Test handling categorical data
  df = DataFrame(a=[1,2], b=categorical(["low","high"]))
  levels!(df.b, ["low", "high"])

  @test isequal(Unmarked.n_levels(df.a), 0)
  @test isequal(Unmarked.n_levels(df.b), 2)
  @test isequal(Unmarked.rep_levels(df.a, 5), repeat([0], 5))
  @test isequal(Unmarked.rep_levels(df.b, 3), ["low","high","low"])

  df[!, :catvar] = categorical(["cat1", "cat1"])
  levels!(df.catvar, ["cat1","cat2","cat3"])
  dfn = Unmarked.add_levels(df)

  @test isequal(nrow(dfn), 3)
  @test isequal(dfn.catvar, levels(dfn.catvar))
  @test isequal(names(dfn), names(df))

  #Fit model with categorical
  ψ_formula = @formula(ψ~elev+forest);
  p_formula = @formula(p~precip+wind);
  β_truth = [0, -0.5, 1.2, -0.2, 0, 0.7];

  umd = simulate(Occu, ψ_formula, p_formula, (100, 5), β_truth);

  ψ_formula = @formula(ψ~elev+forest+catvar);
  umd.site_covs[!,:catvar] = categorical(rand(["high", "med", "low"], 100))
  levels!(umd.site_covs.catvar, ["low","med","high"])

  fit = occu(ψ_formula, p_formula, umd);

  @test isequal(coefnames(fit)[4:5], ["catvar: med", "catvar: high"])

  #Predict
  catvar=categorical(["low","low"])
  levels!(catvar, ["low","med","high"])
  pr_df = DataFrame(elev=[0.5, -0.3], forest=[1,-1], catvar=catvar);
  predict(occupancy(fit), pr_df, interval=true)

end
