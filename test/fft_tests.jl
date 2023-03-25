using DoubleFloats, FFTW, GenericFFT, LinearAlgebra

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
    @test norm(irfft(c, 198) - irfft(map(ComplexF64, c), 198), Inf) < 10eps(Float64)
    @test norm(irfft(c, 199) - irfft(map(ComplexF64, c), 199), Inf) < 10eps(Float64)
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