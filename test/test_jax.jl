using PythonCallChainRules.Jax: JaxFunctionWrapper, jax, stax, dlpack, ispysetup

using Test

#using Zygote
using Random
using PythonCall
using DLPack

if !ispysetup[]
    return
end

@testset "dlpack" begin
    key = jax.random.PRNGKey(0)
    for dims in ((10,), (1, 10), (2, 3, 5), (2, 3, 4, 5))
        xto = jax.random.normal(key, dims)
        xjl = DLPack.wrap(xto, dlpack.to_dlpack)
        @test Tuple(xto.shape) == reverse(size(xjl))
        @test isapprox(jax.numpy.sum(xto).item(), sum(xjl))
    end
end