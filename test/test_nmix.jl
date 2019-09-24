@testset "Nmix GOF" begin
  
  #Test separating cohorts by NA pattern
  Random.seed!(123)
  umd1 = simulate(Nmix, @formula(ψ~elev+forest), @formula(p~wind+precip),
                  (10,3), [0,0,0,0,0,0])
  fit1 = nmix(@formula(ψ~elev), @formula(p~wind), umd1)
  
  #Test fitted
  ft = Unmarked.fitted(fit1)
  @test ft isa Array
  @test length(ft) == 30
  @test round(sum(ft),digits=4) == 23.9460
  
  #Test residuals
  rs = Unmarked.residuals(fit1)
  @test rs isa Array
  @test length(rs) == 30
  @test round(sum(rs), digits=4) == -0.054

  #Test chi-square calc
  @test isequal(round(Unmarked.chisq(fit1),digits=4), 23.0500)
  #with missing values
  #@test isequal(round(Unmarked.mb_chisq(fit2),digits=4), 3.6675)

  #Test parametric bootstrap
  Random.seed!(123)
  gf = gof(fit1)
  @test isequal(length(gf.tstar), 30)
  @test isequal(mean(gf.t0 .< gf.tstar), 0.7000)
  #gf2 = gof(fit2)
  #@test isequal(mean(gf2.t0 .< gf2.tstar), 0.86)
  
end
