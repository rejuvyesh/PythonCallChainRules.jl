using SafeTestsets

@safetestset "PythonCallChainRules.jl" begin
    @safetestset "torch" begin
        include("test_pytorch.jl")
    end
    @safetestset "jax" begin
        include("test_jax.jl")
    end
end
