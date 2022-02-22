using PythonCallChainRules.Torch: TorchModuleWrapper, ispysetup

using Test

using PythonCall
using DLPack

if !ispysetup[]
    return
end

torch = pyimport("torch")
dlpack = pyimport("torch.utils.dlpack")

device = torch.device("cpu")

@testset "dlpack" begin
    for dims in ((10,), (1, 10), (2, 3, 5), (2, 3, 4, 5))
        xto = torch.randn(dims..., device=device)
        xjl = DLPack.wrap(xto, dlpack.to_dlpack)
        @test Tuple(xto.size()) == reverse(size(xjl))
        @test isapprox(pyconvert(Float32, xto.cpu().sum().item()), sum(xjl))
    end
end
