@testset "UmDesign" begin

  y = [[1 0]; [0 1]] 
  sc = DataFrame(a=[1,2], b=[3,4])
  oc = DataFrame(c=[5,6,7,8], d=[7,8,9,10])
  dat = UmData(y, sc, oc)

  umd = Unmarked.UmDesign(:test, @formula(ψ~a+b), 
                          Unmarked.LogitLink(), dat.site_covs)
 
  #Test existing dataframes are unchanged
  @test isequal(sc, DataFrame(a=[1,2], b=[3,4]))
  
  #Check slots
  @test isequal(umd.name, :test)
  @test isequal(umd.formula, @formula(ψ~a+b))
  @test isequal(umd.link, Unmarked.LogitLink())
  @test isequal(umd.data, dat.site_covs)
  @test isequal(umd.coefnames, ["(Intercept)", "a", "b"])
  @test isequal(umd.mat, [[1,1] [1,2] [3,4]])
  @test isnothing(umd.idx)

  #Test getting coefficient indices
  umd2 = Unmarked.UmDesign(:test2, @formula(p~1),
                           Unmarked.LogitLink(), dat.obs_covs)
  Unmarked.add_idx!([umd, umd2])
  @test isequal(umd.idx, 1:3)
  @test isequal(umd2.idx, 4:4)

  #Test transform
  tr = logistic.(umd.mat * [0,0.3,0.5])
  @test isequal(Unmarked.transform(umd, [0,0.3,0.5]), tr)
  
  #Test bad formula
  @test_throws ErrorException Unmarked.UmDesign(:test3, @formula(p~fake), 
                                Unmarked.LogitLink(), dat.obs_covs)

end
