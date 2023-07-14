const AbstractFloats = Union{AbstractFloat,Complex{T} where T<:AbstractFloat}

# We use these type definitions for clarity
const RealFloats = T where T<:AbstractFloat
const ComplexFloats = Complex{T} where T<:AbstractFloat


# The following implements Bluestein's algorithm, following http://www.dsprelated.com/dspbooks/mdft/Bluestein_s_FFT_Algorithm.html
# To add more types, add them in the union of the function's signature.

function generic_fft(x::AbstractVector{T}, region::Integer) where T<:AbstractFloats
    @assert region == 1
    generic_fft(x)
end

function generic_fft!(x::AbstractVector{T}, region::Integer=1) where T<:AbstractFloats
    @assert region == 1
    copyto!(x, generic_fft(x))
end

function generic_fft(x::AbstractVector{T}, region) where {T<:AbstractFloats}
    @assert all(==(1), region)
    generic_fft(x)
end

function generic_fft!(x::AbstractVector{T}, region) where {T<:AbstractFloats}
    @assert all(==(1), region)
    copyto!(x, generic_fft(x))
end

function generic_fft!(x::AbstractMatrix{T}, region::Integer) where T<:AbstractFloats
    if region == 1
        for j in 1:size(x, 2)
            x[:, j] .= generic_fft(@view x[:, j])
        end
    else
        for k in 1:size(x, 1)
            x[k, :] .= generic_fft(@view x[k, :])
        end
    end
    x
end

function generic_fft!(x::AbstractMatrix{T}, region) where T<:AbstractFloats
    for r in region
        generic_fft!(x, r)
    end
    x
end

generic_fft(x::AbstractMatrix{T}, region::Integer) where T<:AbstractFloats = generic_fft!(copy(x), region)

generic_fft(x::AbstractMatrix{T}, region=ntuple(identity, ndims(x))) where T<:AbstractFloats = generic_fft!(copy(x), region)

function generic_fft(x::AbstractVector{T}) where T<:AbstractFloats
    n = length(x)
    ispow2(n) && return generic_fft_pow2(x)
    ks = range(zero(real(T)),stop=n-one(real(T)),length=n)
    Wks = @. cispi(-T(ks^2/n))
    Wksrev = @view Wks[reverse(eachindex(Wks))]
    xq, wq = x.*Wks, conj!([cispi(-T(n)); Wksrev; @view Wks[2:end]])
    return Wks.* @view _conv!(xq,wq)[n+1:2n]
end

generic_bfft(x::AbstractArray{T, N}, region) where {T <: AbstractFloats, N} = conj!(generic_fft(conj(x), region))
generic_bfft!(x::AbstractArray{T, N}, region) where {T <: AbstractFloats, N} = conj!(generic_fft!(conj!(x), region))

_regionscale(x, region::Int) = size(x, region)
_regionscale(x, region) = prod(size.(Ref(x), region))

generic_ifft(x::AbstractArray{T, N}, region) where {T<:AbstractFloats, N} = ldiv!(T(_regionscale(x, region)), conj!(generic_fft(conj(x), region)))
generic_ifft!(x::AbstractArray{T, N}, region) where {T<:AbstractFloats, N} = ldiv!(T(_regionscale(x, region)), conj!(generic_fft!(conj!(x), region)))

generic_rfft(v::AbstractVector{T}, region) where T<:AbstractFloats = generic_fft(v, region)[1:div(length(v),2)+1]
function generic_irfft(v::AbstractVector{T}, n::Integer, region) where T<:ComplexFloats
    @assert length(v) == n>>1 + 1
    r = Vector{T}(undef, n)
    r[1:length(v)]=v
    r[length(v)+1:n]=reverse(conj(v[2:end])[1:n-length(v)])
    real(generic_ifft(r, region))
end
generic_brfft(v::AbstractArray, n::Integer, region) = generic_irfft(v, n, region)*n

function _conv!(u::AbstractVector{T}, v::AbstractVector{T}) where T<:AbstractFloats
    nu = length(u)
    nv = length(v)
    n = nu + nv - 1
    np2 = nextpow(2, n)
    append!(u, zeros(T, np2-nu))
    append!(v, zeros(T, np2-nv))
    y = generic_ifft_pow2(generic_fft_pow2(u).*generic_fft_pow2(v))
    #TODO This would not handle Dual/ComplexDual numbers correctly
    y = T<:Real ? real(y[1:n]) : y[1:n]
end

# This is a Cooley-Tukey FFT algorithm inspired by many widely available algorithms including:
# c_radix2.c in the GNU Scientific Library and four1 in the Numerical Recipes in C.
# However, the trigonometric recurrence is improved for greater efficiency.
# The algorithm starts with bit-reversal, then divides and conquers in-place.
function generic_fft_pow2!(x::AbstractVector{T}) where T<:AbstractFloat
    n,big2=length(x),2one(T)
    nn,j=n÷2,1
    for i=1:2:n-1
        if j>i
            x[j], x[i] = x[i], x[j]
            x[j+1], x[i+1] = x[i+1], x[j+1]
        end
        m = nn
        while m ≥ 2 && j > m
            j -= m
            m = m÷2
        end
        j += m
    end
    logn = 2
    while logn < n
        θ=-big2/logn
        wtemp = sinpi(θ/2)
        wpr, wpi = -2wtemp^2, sinpi(θ)
        wr, wi = one(T), zero(T)
        for m=1:2:logn-1
            for i=m:2logn:n
                j=i+logn
                mixr, mixi = wr*x[j]-wi*x[j+1], wr*x[j+1]+wi*x[j]
                x[j], x[j+1] = x[i]-mixr, x[i+1]-mixi
                x[i], x[i+1] = x[i]+mixr, x[i+1]+mixi
            end
            wr = (wtemp=wr)*wpr-wi*wpi+wr
            wi = wi*wpr+wtemp*wpi+wi
        end
        logn = logn << 1
    end
    return x
end

function generic_fft_pow2(x::AbstractVector{Complex{T}}) where T<:AbstractFloat
    y = interlace_complex(x)
    generic_fft_pow2!(y)
    return deinterlace_complex(y)
end
generic_fft_pow2(x::AbstractVector{T}) where T<:AbstractFloat = generic_fft_pow2(complex(x))

function generic_ifft_pow2(x::AbstractVector{Complex{T}}) where T<:AbstractFloat
    y = interlace_complex(x, -)
    generic_fft_pow2!(y)
    return ldiv!(T(length(x)), deinterlace_complex(y, -))
end

function generic_dct(x::StridedVector{T}, region::Integer) where T<:AbstractFloats
    @assert region == 1
    generic_dct(x)
end

function generic_dct!(x::StridedVector{T}, region::Integer) where T<:AbstractFloats
    @assert region == 1
    copyto!(x, generic_dct(x))
end

function generic_idct(x::StridedVector{T}, region::Integer) where T<:AbstractFloats
    @assert region == 1
    generic_idct(x)
end

function generic_idct!(x::StridedVector{T}, region::Integer) where T<:AbstractFloats
    @assert region == 1
    copyto!(x, generic_idct(x))
end

function generic_dct(x::StridedVector{T}, region::UnitRange{I}) where {T<:AbstractFloats, I<:Integer}
    @assert region == 1:1
    generic_dct(x)
end

function generic_dct!(x::StridedVector{T}, region::UnitRange{I}) where {T<:AbstractFloats, I<:Integer}
    @assert region == 1:1
    copyto!(x, generic_dct(x))
end

function generic_idct(x::StridedVector{T}, region::UnitRange{I}) where {T<:AbstractFloats, I<:Integer}
    @assert region == 1:1
    generic_idct(x)
end

function generic_idct!(x::StridedVector{T}, region::UnitRange{I}) where {T<:AbstractFloats, I<:Integer}
    @assert region == 1:1
    copyto!(x, generic_idct(x))
end

function generic_dct(a::AbstractVector{Complex{T}}) where {T <: AbstractFloat}
    T <: FFTW.fftwNumber && (@warn("Using generic dct for FFTW number type."))
    N = length(a)
    twoN = convert(T,2) * N
    c = generic_fft([a; reverse(a, dims=1)]) # c = generic_fft([a; flipdim(a,1)])
    d = c[1:N]
    d .*= exp.((-im*convert(T, pi)).*(0:N-1)./twoN)
    d[1] = d[1] / sqrt(convert(T, 2))
    lmul!(inv(sqrt(twoN)), d)
end

generic_dct(a::AbstractArray{T}) where {T <: AbstractFloat} = real(generic_dct(complex(a)))

function generic_idct(a::AbstractVector{Complex{T}}) where {T <: AbstractFloat}
    T <: FFTW.fftwNumber && (@warn("Using generic idct for FFTW number type."))
    N = length(a)
    twoN = convert(T,2)*N
    b = a * sqrt(twoN)
    b[1] = b[1] * sqrt(convert(T,2))
    shift = exp.(-im * 2 * convert(T, pi) * (N - convert(T,1)/2) * (0:(2N-1)) / twoN)
    b = [b; 0; -reverse(b[2:end], dims=1)] .* shift # b = [b; 0; -flipdim(b[2:end],1)] .* shift
    c = ifft(b)
    reverse(c[1:N]; dims=1)#flipdim(c[1:N],1)
end

generic_idct(a::AbstractArray{T}) where {T <: AbstractFloat} = real(generic_idct(complex(a)))


# These lines mimick the corresponding ones in FFTW/src/dct.jl, but with
# AbstractFloat rather than fftwNumber.
for f in (:dct, :dct!, :idct, :idct!)
    pf = Symbol("plan_", f)
    @eval begin
        $f(x::AbstractArray{<:AbstractFloats}) = $pf(x) * x
        $f(x::AbstractArray{<:AbstractFloats}, region) = $pf(x, region) * x
    end
end

# dummy plans
abstract type DummyPlan{T} <: Plan{T} end
for P in (:DummyFFTPlan, :DummyiFFTPlan, :DummybFFTPlan, :DummyDCTPlan, :DummyiDCTPlan)
    # All plans need an initially undefined pinv field
    @eval begin
        mutable struct $P{T,inplace,G} <: DummyPlan{T}
            region::G # region (iterable) of dims that are transformed
            pinv::DummyPlan{T}
            $P{T,inplace,G}(region::G) where {T<:AbstractFloats, inplace, G} = new(region)
        end
    end
end
for P in (:DummyrFFTPlan, :DummyirFFTPlan, :DummybrFFTPlan)
    @eval begin
        mutable struct $P{T,inplace,G} <: DummyPlan{T}
            n::Integer
            region::G # region (iterable) of dims that are transformed
            pinv::DummyPlan{T}
            $P{T,inplace,G}(n::Integer, region::G) where {T<:AbstractFloats, inplace, G} = new(n, region)
        end
    end
end

for (Plan,iPlan) in ((:DummyFFTPlan,:DummyiFFTPlan),
                     (:DummyDCTPlan,:DummyiDCTPlan))
   @eval begin
       plan_inv(p::$Plan{T,inplace,G}) where {T,inplace,G} = $iPlan{T,inplace,G}(p.region)
       plan_inv(p::$iPlan{T,inplace,G}) where {T,inplace,G} = $Plan{T,inplace,G}(p.region)
    end
end

# Specific for rfft, irfft and brfft:
plan_inv(p::DummyirFFTPlan{T,inplace,G}) where {T,inplace,G} = DummyrFFTPlan{T,inplace,G}(p.n, p.region)
plan_inv(p::DummyrFFTPlan{T,inplace,G}) where {T,inplace,G} = DummyirFFTPlan{T,inplace,G}(p.n, p.region)



for (Plan,ff,ff!) in ((:DummyFFTPlan,:generic_fft,:generic_fft!),
                      (:DummybFFTPlan,:generic_bfft,:generic_bfft!),
                      (:DummyiFFTPlan,:generic_ifft,:generic_ifft!),
                      (:DummyrFFTPlan,:generic_rfft,:generic_rfft!),
                      (:DummyDCTPlan,:generic_dct,:generic_dct!),
                      (:DummyiDCTPlan,:generic_idct,:generic_idct!))
    @eval begin
        *(p::$Plan{T,true}, x::StridedArray{T,N}) where {T<:AbstractFloats,N} = $ff!(x, p.region)
        *(p::$Plan{T,false}, x::StridedArray{T,N}) where {T<:AbstractFloats,N} = $ff(x, p.region)
        function mul!(C::StridedVector, p::$Plan, x::StridedVector)
            C[:] = $ff(x, p.region)
            C
        end
    end
end

# Specific for irfft and brfft:
*(p::DummyirFFTPlan{T,true}, x::StridedArray{T,N}) where {T<:AbstractFloats,N} = generic_irfft!(x, p.n, p.region)
*(p::DummyirFFTPlan{T,false}, x::StridedArray{T,N}) where {T<:AbstractFloats,N} = generic_irfft(x, p.n, p.region)
function mul!(C::StridedVector, p::DummyirFFTPlan, x::StridedVector)
    C[:] = generic_irfft(x, p.n, p.region)
    C
end
*(p::DummybrFFTPlan{T,true}, x::StridedArray{T,N}) where {T<:AbstractFloats,N} = generic_brfft!(x, p.n, p.region)
*(p::DummybrFFTPlan{T,false}, x::StridedArray{T,N}) where {T<:AbstractFloats,N} = generic_brfft(x, p.n, p.region)
function mul!(C::StridedVector, p::DummybrFFTPlan, x::StridedVector)
    C[:] = generic_brfft(x, p.n, p.region)
    C
end


# We intercept the calls to plan_X(x, region) below.
# In order not to capture any calls that should go to FFTW, we have to be
# careful about the typing, so that the calls to FFTW remain more specific.
# This is the reason for using StridedArray below. We also have to carefully
# distinguish between real and complex arguments.

plan_fft(x::StridedArray{T}, region) where {T <: ComplexFloats} = DummyFFTPlan{T,false,typeof(region)}(region)
plan_fft!(x::StridedArray{T}, region) where {T <: ComplexFloats} = DummyFFTPlan{T,true,typeof(region)}(region)

plan_bfft(x::StridedArray{T}, region) where {T <: ComplexFloats} = DummybFFTPlan{T,false,typeof(region)}(region)
plan_bfft!(x::StridedArray{T}, region) where {T <: ComplexFloats} = DummybFFTPlan{T,true,typeof(region)}(region)

# The ifft plans are automatically provided in terms of the bfft plans above.
# plan_ifft(x::StridedArray{T}, region) where {T <: ComplexFloats} = DummyiFFTPlan{Complex{real(T)},false,typeof(region)}(region)
# plan_ifft!(x::StridedArray{T}, region) where {T <: ComplexFloats} = DummyiFFTPlan{Complex{real(T)},true,typeof(region)}(region)

plan_dct(x::StridedArray{T}, region) where {T <: AbstractFloats} = DummyDCTPlan{T,false,typeof(region)}(region)
plan_dct!(x::StridedArray{T}, region) where {T <: AbstractFloats} = DummyDCTPlan{T,true,typeof(region)}(region)

plan_idct(x::StridedArray{T}, region) where {T <: AbstractFloats} = DummyiDCTPlan{T,false,typeof(region)}(region)
plan_idct!(x::StridedArray{T}, region) where {T <: AbstractFloats} = DummyiDCTPlan{T,true,typeof(region)}(region)

plan_rfft(x::StridedArray{T}, region) where {T <: RealFloats} = DummyrFFTPlan{T,false,typeof(region)}(length(x), region)
plan_brfft(x::StridedArray{T}, n::Integer, region) where {T <: ComplexFloats} = DummybrFFTPlan{T,false,typeof(region)}(n, region)

# A plan for irfft is created in terms of a plan for brfft.
# plan_irfft(x::StridedArray{T}, n::Integer, region) where {T <: ComplexFloats} = DummyirFFTPlan{Complex{real(T)},false,typeof(region)}(n, region)

# These don't exist for now:
# plan_rfft!(x::StridedArray{T}) where {T <: RealFloats} = DummyrFFTPlan{Complex{real(T)},true}()
# plan_irfft!(x::StridedArray{T},n::Integer) where {T <: RealFloats} = DummyirFFTPlan{Complex{real(T)},true}()

function interlace(a::AbstractVector{S},b::AbstractVector{V}) where {S<:Number,V<:Number}
    na=length(a);nb=length(b)
    T=promote_type(S,V)
    if nb≥na
        ret=zeros(T,2nb)
        ret[1:2:1+2*(na-1)]=a
        ret[2:2:end]=b
        ret
    else
        ret=zeros(T,2na-1)
        ret[1:2:end]=a
        if !isempty(b)
            ret[2:2:2+2*(nb-1)]=b
        end
        ret
    end
end

"""
Interlace `a::AbstractVector` with complex entries as

    [(r1,i1), (r2,i2), (r2,i3), ...] -> [r1,i1,r2,i2,r3,i3,...]
    
with `r,i` the real and imaginary part of every element in `a`."""
function interlace_complex(a::AbstractVector{S}, conjfn = identity) where {S<:Complex}
    n = length(a)
    a_interlaced = zeros(real(S),2n)
    @inbounds for (i, ai) in enumerate(a)
        a_interlaced[2i-1] = real(ai)
        a_interlaced[2i] = conjfn(imag(ai))
    end
    return a_interlaced
end

"""Reverse function of interlace_complex."""
function deinterlace_complex(a_interlaced::AbstractVector{S}, conjfn = identity) where {S<:Real}
    n = length(a_interlaced)
    a = zeros(Complex{S}, n ÷ 2)        # deinterlaced vector
    @inbounds for i in eachindex(a)     # ignores last entry if length(a_interlaced) odd
        a[i] = complex(a_interlaced[2i-1], conjfn(a_interlaced[2i]))
    end
    return a
end
