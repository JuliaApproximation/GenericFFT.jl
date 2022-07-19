
using AbstractFFTs, GenericFFT, Test


@test AbstractFFTs.fftfloat(zero(Float16)) isa Float32
@test AbstractFFTs.fftfloat(zero(Float32)) isa Float32
@test AbstractFFTs.fftfloat(zero(Float64)) isa Float64
@test AbstractFFTs.fftfloat(zero(BigFloat)) isa BigFloat
@test AbstractFFTs.fftfloat(zero(Complex{Float16})) isa Complex{Float32}
@test AbstractFFTs.fftfloat(zero(Complex{Float32})) isa Complex{Float32}
@test AbstractFFTs.fftfloat(zero(Complex{Float64})) isa Complex{Float64}
@test AbstractFFTs.fftfloat(zero(Complex{BigFloat})) isa Complex{BigFloat}


include("fft_tests.jl")
