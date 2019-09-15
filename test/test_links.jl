@testset "Link functions" begin
  
  #Logit link
  @test isequal(Unmarked.invlink(0.5, Unmarked.LogitLink()), logistic(0.5))
  @test isequal(Unmarked.invlink([0.3,0.5], Unmarked.LogitLink()),
                logistic.([0.3,0.5]))
  @test isequal(Unmarked.grad(0.5, Unmarked.LogitLink()),
                exp(-0.5)/(1+exp(-0.5))^2)
  @test isequal(Unmarked.grad([0.3,0.5], Unmarked.LogitLink()),
                map(x -> exp(-x)/(1+exp(-x))^2, [0.3,0.5]))

  #Log link
  @test isequal(Unmarked.invlink(0.5, Unmarked.LogLink()), exp(0.5))
  @test isequal(Unmarked.invlink([0.3,0.5], Unmarked.LogLink()),
                exp.([0.3,0.5]))
  @test isequal(Unmarked.grad(0.5, Unmarked.LogLink()), exp(0.5))
  @test isequal(Unmarked.grad([0.3,0.5], Unmarked.LogLink()),
                exp.([0.3,0.5]))

end

