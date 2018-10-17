@testset "primitive" begin

    @testset "Multiply" begin
        input_dim = 10; output_dim = 2
        m = Multiply(input=input_dim, output=output_dim, atype=_test_atype)
        N = rand(1:10)
        x = _test_atype(zeros(input_dim, N))
        y1 = m(x)
        y2 = _test_atype(zeros(output_dim, N))
        @test y1 == y2
    end

    
end


