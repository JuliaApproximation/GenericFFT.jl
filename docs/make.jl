using Documenter, GenericFFT

makedocs(
	doctest = false,
	clean = true,
	format = Documenter.HTML(),
	sitename = "GenericFFT.jl",
	authors = "volunteers wanted",
	pages = Any[
			"Home" => "index.md"
	]
)


deploydocs(
    repo   = "github.com/JuliaApproximation/GenericFFT.jl.git"
)
