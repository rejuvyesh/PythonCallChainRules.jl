module Jax

using PythonCall: Py, pynew, pyimport, pycopy!
using ChainRulesCore
using DLPack
using Functors: fmap
using Adapt

using ..PythonCallChainRules: PyAdaptor

const inspect = pynew()
const jax = pynew()
const dlpack = pynew()
const stax = pynew()
const numpy = pynew()

const ispysetup = Ref{Bool}(false)

struct JaxFunctionWrapper
    jaxfn::Py
end

function (wrap::JaxFunctionWrapper)(args...; kwargs...)
    # TODO: handle multiple outputs
    out = (wrap.jaxfn(fmap(x->DLPack.share(x, Py, dlpack.from_dlpack), args)...))
    return DLPack.wrap(out, dlpack.to_dlpack)
end

function ChainRulesCore.rrule(wrap::JaxFunctionWrapper, args...; kwargs...)
    T = typeof(first(args))
    project = ProjectTo(args)
    jax_primal, jax_vjpfun = jax.vjp(wrap.jaxfn, fmap(x->DLPack.share(x, Py, dlpack.from_dlpack), args)...; kwargs...)
    function JaxFunctionWrapper_pullback(Δ)
        cΔ = Adapt.adapt(PyAdaptor{T}, Δ)
        dlΔ = DLPack.share(cΔ, Py, dlpack.from_dlpack)
        tangent_vals = fmap(x->DLPack.wrap(x, dlpack.to_dlpack), jax_vjpfun(dlΔ))
        return (NoTangent(), project(tangent_vals)...)
    end
    return (DLPack.wrap(jax_primal, dlpack.to_dlpack)), JaxFunctionWrapper_pullback
end

function __init__()
    try
        pycopy!(jax, pyimport("jax"))
        pycopy!(numpy, pyimport("numpy"))
        pycopy!(stax, pyimport("jax.example_libraries.stax"))
        pycopy!(inspect, pyimport("inspect"))
        ispysetup[] = true
    catch err
        @warn """PythonCallChainRules.jl has failed to import jax from Python.
                 Please make sure these are installed. 
                 methods of this package.
        """
        @debug err   
        ispysetup[] = false           
    end
end

end