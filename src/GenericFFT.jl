module GenericFFT

using Reexport

@reexport using AbstractFFTs
@reexport using FFTW

import Base: *

import AbstractFFTs: Plan, ScaledPlan,
                     fft, ifft, bfft, fft!, ifft!, bfft!, rfft, irfft, brfft,
                     plan_fft, plan_ifft, plan_bfft, plan_fft!, plan_ifft!,
                     plan_bfft!, plan_rfft, plan_irfft, plan_brfft,
                     fftshift, ifftshift, rfft_output_size, brfft_output_size,
                     plan_inv, normalization

import FFTW: dct, dct!, idct, idct!, plan_dct!, plan_idct!,
             plan_dct, plan_idct, fftwNumber

import LinearAlgebra: mul!, lmul!, ldiv!

# We override these for AbstractFloat, so that conversion from reals to
# complex numbers works for any AbstractFloat (instead of only BlasFloat's)
AbstractFFTs.complexfloat(x::StridedArray{Complex{<:AbstractFloat}}) = x
AbstractFFTs.realfloat(x::StridedArray{<:Real}) = x
# We override this one in order to avoid throwing an error that the type is
# unsupported (as defined in AbstractFFTs)
AbstractFFTs._fftfloat(::Type{T}) where {T <: AbstractFloat} = T
# We also avoid any conversion of types that are already AbstractFloat
# (since AbstractFFTs calls float(x) by default, which might change types)
AbstractFFTs.fftfloat(x::AbstractFloat) = x
# for compatibility with AbstractFFTs
AbstractFFTs.fftfloat(x::Float16) = Float32(x)


include("fft.jl")

end # module
