# TODO: Add backward testing after new-gradcheck in AutoGrad
@testset "primitive" begin

    @testset "Multiply" begin

        m = Multiply(input=_INPUTdim, output=_OUTPUTdim, atype=_test_atype)
        B = rand(1:10)

        x1   = _test_atype(zeros(_INPUTdim, B))
        y1_1 = m(x1)
        y1_2 = _test_atype(zeros(_OUTPUTdim, B))
        @test y1_1 == y1_2        

        x2   = _test_atype(randn(_INPUTdim, B))
        y2_1 = m(x2)
        y2_2 = m.w * x2
        @test y2_1 == y2_2
    end

    @testset "Embed" begin
        x1 = rand(1:_INPUTdim, 5)
        m  = Embed(input=_INPUTdim, output=_OUTPUTdim, atype=_test_atype)
        y1 = m(x1)
        y2 = hcat([m.w[:,i] for i in x1]...)

        @test y1 == y2
    end

    @testset "BatchMul" begin
        m  = BatchMul(input=_INPUTdim, output=_OUTPUTdim, atype=_test_atype)
        B  = rand(1:20); T = rand(1:4)

        x1 = _test_atype(randn(_INPUTdim, B, T))
        y1 = m(x1, flatten=true)
        # put batched instances into the columns
        x1_flat = hcat([ x1[:,:,i] for i  in 1:T ]...)
        y2 = m.w * x1_flat
        @test y1 == y2

        x2   = _test_atype(randn(_INPUTdim, B, T, 3))
        y2_1 = m(x2, flatten=true)
        x2_flat = reshape(x2, _INPUTdim, B*T*3)
        y2_2 = m.w * x2_flat
        @test y2_2 == y2_1
    end

    @testset "Linear" begin
        m = Linear(input=_INPUTdim, output=_OUTPUTdim, atype=_test_atype)
        x1 = _test_atype(zeros(_INPUTdim, 1))
        y1 = m(x1)
        y2 = _test_atype(zeros(_OUTPUTdim, 1))
        @test y1 == y2

        m = Linear(input=_INPUTdim, output=_OUTPUTdim, atype=_test_atype, binit=ones, winit=randn)
        y1_2 = m(x1)
        y2_2 = _test_atype(ones(_OUTPUTdim, 1))
        @test y1_2 == y2_2
    end
    
end


