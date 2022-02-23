module Torch

using PythonCall: Py, pynew, pycopy!, @pyeval, @pyexec

using ChainRulesCore
using DLPack
using Functors: @functor
using Adapt


using ..PythonCallChainRules: PyAdaptor

const inspect = pynew()
const torch = pynew()
const functorch = pynew()
const dlpack = pynew()

const ispysetup = Ref{Bool}(false)

struct TorchModuleWrapper
    torch_stateless_module::Py
    dtype::Py
    params::Tuple
    buffers::Py
end

@functor TorchModuleWrapper (params,)

Base.show(io::IO, f::TorchModuleWrapper) = print(io, f.torch_stateless_module, " ", f.dtype, " ", " params size=", size.(f.params))

Base.length(f::TorchModuleWrapper) = length(f.params)
Base.iterate(f::TorchModuleWrapper) = iterate(f.params)
Base.iterate(f::TorchModuleWrapper, state) = iterate(f.params, state)

function TorchModuleWrapper(torch_module)
    pyconvert(Bool, pyisinstance(torch_module, torch.nn.Module)) || error("Not a torch.nn.Module")
    funmod, params, buffers = functorch.make_functional_with_buffers(torch_module)
    dtype = params[0].dtype
    jlparams = pyconvert(Tuple, map(x->DLPack.wrap(x, dlpack.to_dlpack), params))
    return TorchModuleWrapper(funmod, dtype, jlparams, buffers)
end

function (wrap::TorchModuleWrapper)(args...; kwargs...)
    # TODO: handle multiple outputs
    params = wrap.params
    tensor_out = wrap.torch_stateless_module((map(x -> DLPack.share(x, Py, dlpack.from_dlpack).requires_grad_(true), params)),
        wrap.buffers, map(x -> DLPack.share(x, Py, dlpack.from_dlpack), args)...; kwargs...)
    res = DLPack.wrap(tensor_out, dlpack.to_dlpack)
    return res
end

function ChainRulesCore.rrule(wrap::TorchModuleWrapper, args...; kwargs...)
    T = typeof(first(args))
    params = wrap.params
    torch_primal, torch_vjpfun = functorch.vjp(@pyeval("buffer_implicit")(wrap.torch_stateless_module, wrap.buffers), Tuple(map(x -> DLPack.share(x, Py, dlpack.from_dlpack).requires_grad_(true), params)),
        map(x -> DLPack.share(x, Py, dlpack.from_dlpack).requires_grad_(true), args)...; kwargs...)
    project = ProjectTo(args)
    function TorchModuleWrapper_pullback(Δ)
        torch_tangent_vals = torch_vjpfun(DLPack.share(Adapt.adapt(PyAdaptor{T}, Δ), Py, dlpack.from_dlpack))
        jlparams_tangents = map(x -> (DLPack.wrap(x, dlpack.to_dlpack)), torch_tangent_vals[1])
        args_tangents = project(map(x -> (DLPack.wrap(x, dlpack.to_dlpack)), torch_tangent_vals[2:end]))
        return (Tangent{TorchModuleWrapper}(; torch_stateless_module = NoTangent(), dtype = NoTangent(), params = jlparams_tangents, buffers = NoTangent()), args_tangents...)
    end
    res = DLPack.wrap(torch_primal, dlpack.to_dlpack)
    return res, TorchModuleWrapper_pullback
end



function __init__()
    try
        pycopy!(torch, pyimport("torch"))
        pycopy!(dlpack, pyimport("torch.utils.dlpack"))
        pycopy!(functorch, pyimport("functorch"))
        pycopy!(inspect, pyimport("inspect"))
        ispysetup[] = true
        @pyexec """
        global buffer_implicit
        def buffer_implicit(fn, buffers):
            def newfn(params, inputs):
                return fn(params, buffers, inputs)
            
            return newfn
        """        
    catch err
        @warn """PythonCallChainRules.jl has failed to import torch and functorch from Python.
                 Please make sure these are installed. 
        """
        @debug err
        ispysetup[] = false
        #rethrow(err)        
    end
end

end