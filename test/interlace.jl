@testset "interlace complex vectors" begin
    for F in (Float16, Float32, Float64)
        for n in (4,7,8,9,12,14,16)

            # deinterlace(interlace)
            v = rand(Complex{F},n)
            v_interlaced = GenericFFT.interlace_complex(v)
            v2 = GenericFFT.deinterlace_complex(v_interlaced)
            @test v == v2

            # and interlace(deinterlace)
            v_interlaced = rand(F,2n)
            v = GenericFFT.deinterlace_complex(v_interlaced)
            v_interlaced2 = GenericFFT.interlace_complex(v)
            @test v_interlaced == v_interlaced2

            # odd interlaced vectors (shouldn't be produced and will ignore last element)
            v_interlaced = rand(F,2n+1)
            v = GenericFFT.deinterlace_complex(v_interlaced)   # [end] ignored here
            v_interlaced2 = GenericFFT.interlace_complex(v)
            @test v_interlaced[1:end-1] == v_interlaced2
        end
    end
end