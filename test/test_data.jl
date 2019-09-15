@testset "UmData" begin

  y = [[1 0]; [0 1]; [1 1]] 
  sc = DataFrame(a=[1,2,3], b=[3,4,5])
  oc = DataFrame(c=collect(1:6), d=collect(1:6))
  dat = UmData(y, sc, oc)
  @test isequal(dat.y, y)
  @test isequal(dat.site_covs, sc)
  @test isequal(dat.obs_covs, oc)

  yna = [[missing 0]; [0 1]; [1 1]]
  dat2 = UmData(yna, sc, oc)
  @test isequal(dat2.y, yna)

  scna = DataFrame(a=[1,missing],b=[3,4])
  @test_throws ErrorException Unmarked.check_data(scna)
  
  #Test subsetting
  ds1 = dat[1]
  @test isequal(size(ds1.y), (1,2))
  @test isequal(ds1.y[1,:], y[1,:])
  @test isequal(ds1.site_covs, sc[1:1,:])
  @test isequal(ds1.obs_covs, oc[1:2,:])

  ds2 = dat[2:3]
  ds3 = dat[[2,3]]
  @test isequal(ds2.y, ds3.y)
  @test isequal(ds2.y, y[2:3,:])
  @test isequal(ds2.site_covs, sc[2:3,:])
  @test isequal(ds2.obs_covs, oc[3:6,:])

end
