# GenericFFT.jl

[![Build Status](https://github.com/JuliaApproximation/GenericFFT.jl/workflows/CI/badge.svg)](https://github.com/JuliaApproximation/GenericFFT.jl/actions?query=workflow%3ACI) [![codecov](https://codecov.io/gh/JuliaApproximation/GenericFFT.jl/branch/main/graph/badge.svg?token=TSiBjCYqzb)](https://codecov.io/gh/JuliaApproximation/GenericFFT.jl) [![](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaApproximation.github.io/GenericFFT.jl/stable) [![](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaApproximation.github.io/GenericFFT.jl/dev)

`GenericFFT.jl` implements a Fast Fourier Transform for generic floating point number types.

The transforms provided have the right computational complexity. However, for the time being, the implementations are crude and far from optimal. Please consider contributing improvements to the package and filing issues for missing functionality.

## Installation

Installation is straightforward:
```julia
pkg> add GenericFFT

julia> using GenericFFT
```

## Usage for high-precision FFTs

The main reason for using `GenericFFT` is high-precision calculations. For example:
```julia
julia> using GenericFFT

julia> fft(rand(Complex{BigFloat}, 2))
2-element Vector{Complex{BigFloat}}:
 0.8071607526060331187983248443648586158893950448440777116281652091029932491471374 + 1.058204007570364569492040922226041865648762106924785198005758849420721686004251im
 0.3195699335469630499276014344115859560577992018210584550701583748039853943955188 + 0.196737316420669631800810230623687615407691727320510522950346182385847637522683im
```

Other packages provide high-precision floating point numbers, including [QuadMath.jl](https://github.com/JuliaMath/Quadmath.jl) and [DoubleFloats.jl](https://github.com/JuliaMath/DoubleFloats.jl).
```julia
julia> using GenericFFT, DoubleFloats

julia> fft(rand(Double64, 2))
2-element Vector{Complex{Double64}}:
 0.4026739024263829 + 0.0im
 0.3969515892883767 + 0.0im
 ```

## History

The code in this package was developed in the [FastTransforms.jl](https://github.com/JuliaApproximation/FastTransforms.jl) package by Mikael Slevinsky. The code was moved to a separate package in July 2022 from [this file](https://github.com/JuliaApproximation/FastTransforms.jl/blob/3bd5a9a2cf744fc26418fe999bbb151b5ccc6634/src/fftBigFloat.jl).
