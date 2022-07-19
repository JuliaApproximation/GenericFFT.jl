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

include("fft.jl")

end # module
