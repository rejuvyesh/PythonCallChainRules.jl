using PythonCallChainRules.Torch: TorchModuleWrapper, torch, functorch, dlpack, ispysetup

using Test

using PythonCall
using DLPack

if !ispysetup[]
    return
end

device = torch.device("cpu")

@testset "dlpack" begin
    for dims in ((10,), (1, 10), (2, 3, 5), (2, 3, 4, 5))
        xto = torch.randn(dims..., device=device)
        xjl = DLPack.wrap(xto, dlpack.to_dlpack)
        @test Tuple(xto.size()) == reverse(size(xjl))
        @test isapprox(sum(xto.cpu().numpy()), sum(xjl))
    end
end
