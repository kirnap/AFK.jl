# TODO: Add backward testing after new-gradcheck in AutoGrad
@testset "primitive" begin

    @testset "Multiply" begin

        m = Multiply(input=_INPUTdim, output=_OUTPUTdim, atype=_test_atype)
        N = rand(1:10)

        x1   = _test_atype(zeros(_INPUTdim, N))
        y1_1 = m(x1)
        y1_2 = _test_atype(zeros(_OUTPUTdim, N))
        
        x2   = _test_atype(randn(_INPUTdim, N))
        y2_1 = m(x2)
        y2_2 = m.w * x2

        @test y1_1 == y1_2
        @test y2_1 == y2_2
    end

    @testset "Embed" begin
        x1 = rand(1:_INPUTdim, 5)
        m  = Embed(input=_INPUTdim, output=_OUTPUTdim, atype=_test_atype)
        y1 = m(x1)
        y2 = hcat([m.w[:,i] for i in x1]...)

        @test y1 == y2
    end
    
end


