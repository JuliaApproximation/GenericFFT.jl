using DoubleFloats, FFTW, GenericFFT, LinearAlgebra
import GenericFFT: generic_fft, generic_fft!

function test_basic_functionality()
    c = randn(ComplexF16, 20)
    p = plan_fft(c)
    @test inv(p) * (p * c) ≈ c

    c = randn(ComplexF16, 20)
    pinpl = plan_fft!(c)
    @test inv(pinpl) * (pinpl * c) ≈ c
end

# Test FFT and DCT functionality for float of type T
function test_fft_dct(T)
    c = collect(range(-one(T),stop=one(T),length=16))
    @test norm(fft(c) - fft(Float64.(c))) < 3Float64(norm(c))*eps(Float64)
    @test norm(ifft(c) - ifft(Float64.(c))) < 3Float64(norm(c))*eps(Float64)

    c = collect(range(-one(T),stop=one(T),length=201))
    @test norm(ifft(fft(c))-c) < 200norm(c)eps(T)

    s = one(T) ./ (1:10)
    s64 = Float64.(s)
    # @test Float64.(conv(s, s)) ≈ conv(s64, s64)
    @test s == one(T) ./ (1:10) #67, ensure conv doesn't overwrite input
    @test all(s64 .=== Float64.(one(T) ./ (1:10)))

    p = plan_dct(c)
    @test norm(GenericFFT.generic_dct(c) - p*c) == 0

    pli = plan_idct!(c)
    @test norm(pli*dct(c) - c) < 1000norm(c)*eps(T)

    @test norm(dct(c)-dct(map(Float64,c)),Inf) < 10eps(Float64)

    cc = cis.(c)
    @test norm(dct(cc)-dct(map(Complex{Float64},cc)),Inf) < 10eps(Float64)

    c = rand(T,100) + im*rand(T,100)
    @test norm(dct(c)-dct(map(ComplexF64,c)),Inf) < 10eps(Float64)
    @test norm(idct(c)-idct(map(ComplexF64,c)),Inf) < 10eps(Float64)
    @test norm(idct(dct(c))-c,Inf) < 1000eps(T)
    @test norm(dct(idct(c))-c,Inf) < 1000eps(T)

    @test_throws AssertionError irfft(c, 197)
    @test norm(irfft(c, 198) - irfft(map(ComplexF64, c), 198), Inf) < 100eps(Float64)
    @test norm(irfft(c, 199) - irfft(map(ComplexF64, c), 199), Inf) < 100eps(Float64)
    @test_throws AssertionError irfft(c, 200)
end

# Make sure we don't accidentally hijack any FFTW plans
function test_fftw()
    for T in (Float32, Float64)
        @test plan_fft(rand(BigFloat,10)) isa GenericFFT.DummyPlan
        @test plan_fft(rand(BigFloat,10), 1:1) isa GenericFFT.DummyPlan
        @test plan_fft(rand(Complex{BigFloat},10)) isa GenericFFT.DummyPlan
        @test plan_fft(rand(Complex{BigFloat},10), 1:1) isa GenericFFT.DummyPlan
        @test plan_fft!(rand(Complex{BigFloat},10)) isa GenericFFT.DummyPlan
        @test plan_fft!(rand(Complex{BigFloat},10), 1:1) isa GenericFFT.DummyPlan
        @test !( plan_fft(rand(T,10)) isa GenericFFT.DummyPlan )
        @test !( plan_fft(rand(T,10), 1:1) isa GenericFFT.DummyPlan )
        @test !( plan_fft(rand(Complex{T},10)) isa GenericFFT.DummyPlan )
        @test !( plan_fft(rand(Complex{T},10), 1:1) isa GenericFFT.DummyPlan )
        @test !( plan_fft!(rand(Complex{T},10)) isa GenericFFT.DummyPlan )
        @test !( plan_fft!(rand(Complex{T},10), 1:1) isa GenericFFT.DummyPlan )

        @test plan_ifft(rand(T,10)) isa FFTW.ScaledPlan
        @test plan_ifft(rand(T,10), 1:1) isa FFTW.ScaledPlan
        @test plan_ifft(rand(Complex{T},10)) isa FFTW.ScaledPlan
        @test plan_ifft(rand(Complex{T},10), 1:1) isa FFTW.ScaledPlan
        @test plan_ifft!(rand(Complex{T},10)) isa FFTW.ScaledPlan
        @test plan_ifft!(rand(Complex{T},10), 1:1) isa FFTW.ScaledPlan

        @test plan_bfft(rand(BigFloat,10)) isa GenericFFT.DummyPlan
        @test plan_bfft(rand(BigFloat,10), 1:1) isa GenericFFT.DummyPlan
        @test plan_bfft(rand(Complex{BigFloat},10)) isa GenericFFT.DummyPlan
        @test plan_bfft(rand(Complex{BigFloat},10), 1:1) isa GenericFFT.DummyPlan
        @test plan_bfft!(rand(Complex{BigFloat},10)) isa GenericFFT.DummyPlan
        @test plan_bfft!(rand(Complex{BigFloat},10), 1:1) isa GenericFFT.DummyPlan
        @test !( plan_bfft(rand(T,10)) isa GenericFFT.DummyPlan )
        @test !( plan_bfft(rand(T,10), 1:1) isa GenericFFT.DummyPlan )
        @test !( plan_bfft(rand(Complex{T},10)) isa GenericFFT.DummyPlan )
        @test !( plan_bfft(rand(Complex{T},10), 1:1) isa GenericFFT.DummyPlan )
        @test !( plan_bfft!(rand(Complex{T},10)) isa GenericFFT.DummyPlan )
        @test !( plan_bfft!(rand(Complex{T},10), 1:1) isa GenericFFT.DummyPlan )

        @test plan_dct(rand(BigFloat,10)) isa GenericFFT.DummyPlan
        @test plan_dct(rand(BigFloat,10), 1:1) isa GenericFFT.DummyPlan
        @test plan_dct(rand(Complex{BigFloat},10)) isa GenericFFT.DummyPlan
        @test plan_dct(rand(Complex{BigFloat},10), 1:1) isa GenericFFT.DummyPlan
        @test plan_dct!(rand(Complex{BigFloat},10)) isa GenericFFT.DummyPlan
        @test plan_dct!(rand(Complex{BigFloat},10), 1:1) isa GenericFFT.DummyPlan
        @test !( plan_dct(rand(T,10)) isa GenericFFT.DummyPlan )
        @test !( plan_dct(rand(T,10), 1:1) isa GenericFFT.DummyPlan )
        @test !( plan_dct(rand(Complex{T},10)) isa GenericFFT.DummyPlan )
        @test !( plan_dct(rand(Complex{T},10), 1:1) isa GenericFFT.DummyPlan )
        @test !( plan_dct!(rand(Complex{T},10)) isa GenericFFT.DummyPlan )
        @test !( plan_dct!(rand(Complex{T},10), 1:1) isa GenericFFT.DummyPlan )

        @test plan_idct(rand(BigFloat,10)) isa GenericFFT.DummyPlan
        @test plan_idct(rand(BigFloat,10), 1:1) isa GenericFFT.DummyPlan
        @test plan_idct(rand(Complex{BigFloat},10)) isa GenericFFT.DummyPlan
        @test plan_idct(rand(Complex{BigFloat},10), 1:1) isa GenericFFT.DummyPlan
        @test plan_idct!(rand(Complex{BigFloat},10)) isa GenericFFT.DummyPlan
        @test plan_idct!(rand(Complex{BigFloat},10), 1:1) isa GenericFFT.DummyPlan
        @test !( plan_idct(rand(T,10)) isa GenericFFT.DummyPlan )
        @test !( plan_idct(rand(T,10), 1:1) isa GenericFFT.DummyPlan )
        @test !( plan_idct(rand(Complex{T},10)) isa GenericFFT.DummyPlan )
        @test !( plan_idct(rand(Complex{T},10), 1:1) isa GenericFFT.DummyPlan )
        @test !( plan_idct!(rand(Complex{T},10)) isa GenericFFT.DummyPlan )
        @test !( plan_idct!(rand(Complex{T},10), 1:1) isa GenericFFT.DummyPlan )

        @test plan_rfft(rand(BigFloat,10)) isa GenericFFT.DummyPlan
        @test plan_rfft(rand(BigFloat,10), 1:1) isa GenericFFT.DummyPlan
        @test plan_brfft(rand(Complex{BigFloat},10), 19) isa GenericFFT.DummyPlan
        @test plan_brfft(rand(Complex{BigFloat},10), 19, 1:1) isa GenericFFT.DummyPlan
        @test !( plan_rfft(rand(T,10)) isa GenericFFT.DummyPlan )
        @test !( plan_rfft(rand(T,10), 1:1) isa GenericFFT.DummyPlan )
        @test !( plan_brfft(rand(Complex{T},10), 19) isa GenericFFT.DummyPlan )
        @test !( plan_brfft(rand(Complex{T},10), 19, 1:1) isa GenericFFT.DummyPlan )

        # check that GenericFFT and FFTW plans have the same parametric type
        @test plan_rfft(rand(Float16,10)) isa AbstractFFTs.Plan{Float16}
        @test plan_rfft(rand(Float64,10)) isa AbstractFFTs.Plan{Float64}
        
        @test plan_brfft(rand(Complex{Float16},10),19) isa AbstractFFTs.Plan{Complex{Float16}}
        @test plan_brfft(rand(Complex{Float64},10),19) isa AbstractFFTs.Plan{Complex{Float64}}
    end
end

@testset "Generic FFT and DCT" begin
    test_basic_functionality()
    test_fft_dct(BigFloat)
    test_fft_dct(Double64)
end

@testset "Test FFTW compatibility" begin
    test_fftw()
end

@testset "inv DFT" begin
    x = big.(randn(10,3) .+ im .* randn(10,3))
    p = plan_fft!(x, 1)
    p_i = plan_ifft!(x, 1)
    @test ComplexF64.(p*copy(x)) ≈ fft(ComplexF64.(x), 1)
    @test ComplexF64.(p_i * copy(x)) ≈ ifft(ComplexF64.(x), 1)
    @test ComplexF64.(p \ copy(x)) ≈ ifft(ComplexF64.(x), 1)

    p = plan_fft!(x, 2)
    p_i = plan_ifft!(x, 2)
    @test ComplexF64.(p*copy(x)) ≈ fft(ComplexF64.(x), 2)
    @test ComplexF64.(p_i * copy(x)) ≈ ifft(ComplexF64.(x), 2)
    @test ComplexF64.(p \ copy(x)) ≈ ifft(ComplexF64.(x), 2)

    p = plan_fft!(x)
    p_i = plan_ifft!(x)
    @test ComplexF64.(p*copy(x)) ≈ fft(ComplexF64.(x))
    @test ComplexF64.(p_i * copy(x)) ≈ ifft(ComplexF64.(x))
    @test ComplexF64.(p \ copy(x)) ≈ ifft(ComplexF64.(x))
end

@testset "generic_fft" begin
    x = randn(5) .+ randn(5)im
    x̃ = copy(x)
    @test generic_fft(x) ≈ generic_fft(x, 1:1) ≈ generic_fft(x, (1,)) ≈ generic_fft!(x̃) ≈ fft(x)
    @test x̃ ≈ fft(x)

    X = randn(5,6) .+ randn(5,6)im
    X̃ = copy(X)
    @test generic_fft(X,1) ≈ generic_fft(X, 1:1) ≈ generic_fft!(X̃,1) ≈ fft(X,1)
    @test X̃ ≈ fft(X,1)
    X̃ = copy(X)
    @test generic_fft(X,2) ≈ generic_fft(X, 2:2) ≈ generic_fft!(X̃,2) ≈ fft(X,2)
    @test X̃ ≈ fft(X,2)
    X̃ = copy(X)
    @test generic_fft(X) ≈ generic_fft(X, 1:2) ≈ generic_fft!(X̃) ≈ fft(X)
    @test X̃ ≈ fft(X)

    X = randn(ComplexF64, 5, 6, 7)
    for d in 1:3
        X̃ = copy(X)
        @test generic_fft(X,d) ≈ generic_fft(X, d:d) ≈ generic_fft!(X̃, (d,)) ≈ fft(X,d)
        @test X̃ ≈ fft(X,d)
    end
    X1 = copy(X)
    X2 = copy(X)
    @test generic_fft(X) ≈ generic_fft(X, 1:ndims(X)) ≈ generic_fft!(X1, 1:ndims(X1)) ≈ generic_fft!(X2) ≈ fft(X)
    @test generic_fft(X, (1,3)) ≈ fft(X, (1,3))
    @test generic_fft(X, (2,3)) ≈ fft(X, (2,3))
    @test generic_fft(X, (1,2)) ≈ fft(X, (1,2))
    @test generic_fft(X, (2,1)) ≈ fft(X, (2,1))
    @test X1 ≈ fft(X)
    @test X2 ≈ fft(X)

    N = 32
    A1 = randn(ComplexF64, N)
    @allocations generic_fft!(A1)  # compile
    @test 0 == @allocations generic_fft!(A1)

    A2 = randn(ComplexF64, N, N, N)
    @allocations generic_fft!(A2)  # compile
    @test N+150 > @allocations generic_fft!(A2)  # a few allocations is OK
end

@testset "Batched rfft/irfft" begin
    for T in (Float64, BigFloat)
        X = randn(T, 10, 6)
        
        Y1 = rfft(X, 1) # Dimension 1
        @test size(Y1) == (10÷2+1, 6)
        for j in 1:6
            @test Y1[:, j] ≈ rfft(X[:, j])
        end
        @test irfft(Y1, 10, 1) ≈ X
        
        Y2 = rfft(X, 2)  # Dimension 2
        @test size(Y2) == (10, 6÷2+1)
        for i in 1:10
            @test Y2[i, :] ≈ rfft(X[i, :])
        end
        @test irfft(Y2, 6, 2) ≈ X

        Y12 = rfft(X, (1, 2)) # 2D RFFT
        @test size(Y12) == (10÷2+1, 6)
        @test Y12 ≈ fft(rfft(X, 1), 2)
        @test irfft(Y12, 10, (1, 2)) ≈ X

        p1 = plan_rfft(X, 1) # Plans
        @test p1 * X ≈ rfft(X, 1)
        @test inv(p1) * (p1 * X) ≈ X

        p2 = plan_rfft(X, 2)
        @test p2 * X ≈ rfft(X, 2)
        @test inv(p2) * (p2 * X) ≈ X
    end


    for n in (7, 11) # Test a few odd lengths
        X = randn(BigFloat, n, 4)
        Y = rfft(X, 1)
        @test size(Y) == (n÷2+1, 4)
        @test irfft(Y, n, 1) ≈ X
    end

    data = randn(BigFloat, 10, 10)
    v = view(data, 1:8, 1:6)
    @test rfft(v, 1) ≈ rfft(collect(v), 1)
    @test irfft(rfft(v, 1), 8, 1) ≈ v

    X3 = randn(BigFloat, 4, 10, 4) # Test 3D Batched
    
    Y3 = rfft(X3, 2) # Transform along dimension 2
    @test size(Y3) == (4, 10÷2+1, 4)
    for i in 1:4, k in 1:4
        @test Y3[i, :, k] ≈ rfft(X3[i, :, k])
    end
    @test irfft(Y3, 10, 2) ≈ X3

    X4 = randn(BigFloat, 3, 3, 3, 3) # Test 4D
    Y4 = rfft(X4, (1, 2, 3)) # RFFT over first 3 dimensions
    @test size(Y4) == (3÷2+1, 3, 3, 3)
    @test irfft(Y4, 3, (1, 2, 3)) ≈ X4
    @test irfft(rfft(X4, (1, 2, 3, 4)), 3, (1, 2, 3, 4)) ≈ X4

    X_single = randn(BigFloat, 10, 1)
    @test rfft(X_single, 1) ≈ rfft(vec(X_single))
    @test irfft(rfft(X_single, 1), 10, 1) ≈ X_single

    X_br = randn(BigFloat, 10, 6)
    Y_br = rfft(X_br, (1, 2))
    # brfft should be irfft * (10 * 6)
    @test brfft(Y_br, 10, (1, 2)) ≈ irfft(Y_br, 10, (1, 2)) * 60
end

@testset "Real-input generic_fft coverage" begin
    X = randn(BigFloat, 8, 8)
    @test GenericFFT.generic_fft(X, 1) ≈ fft(complex(X), 1)
    @test GenericFFT.generic_fft(X, (1, 2)) ≈ fft(complex(X))
end
