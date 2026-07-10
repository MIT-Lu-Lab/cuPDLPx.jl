import MathOptInterface as MOI

const Lib = CuPDLPx.LibCuPDLPx

MOI.Utilities.@product_of_sets(
    _LPProductOfSets,
    MOI.EqualTo{T},
    MOI.GreaterThan{T},
    MOI.LessThan{T},
    MOI.Interval{T},
)

const OptimizerCache = MOI.Utilities.GenericModel{
    Cdouble,
    MOI.Utilities.ObjectiveContainer{Cdouble},
    MOI.Utilities.VariablesContainer{Cdouble},
    MOI.Utilities.MatrixOfConstraints{
        Cdouble,
        MOI.Utilities.MutableSparseMatrixCSC{
            Cdouble,
            Cint,
            MOI.Utilities.ZeroBasedIndexing,
        },
        MOI.Utilities.Hyperrectangle{Cdouble},
        _LPProductOfSets{Cdouble},
    },
}

Base.show(io::IO, ::Type{OptimizerCache}) = print(io, "CuPDLPx.OptimizerCache")

const BOUND_SETS = Union{
    MOI.GreaterThan{Float64},
    MOI.LessThan{Float64},
    MOI.EqualTo{Float64},
    MOI.Interval{Float64},
}

mutable struct Optimizer <: MOI.AbstractOptimizer
    result::Union{Nothing,Lib.cupdlpx_result_t}
    native_result_ptr::Ptr{Lib.cupdlpx_result_t}
    native_problem_ptr::Ptr{Lib.lp_problem_t}
    parameters::Lib.pdhg_parameters_t
    sets::Union{Nothing,_LPProductOfSets{Cdouble}}
    max_sense::Bool
    silent::Bool

    c::Vector{Cdouble}
    obj_const::Vector{Cdouble}
    row_lower::Vector{Cdouble}
    row_upper::Vector{Cdouble}
    var_lower::Vector{Cdouble}
    var_upper::Vector{Cdouble}

    A_colptr::Vector{Cint}
    A_rowval::Vector{Cint}
    A_nzval::Vector{Cdouble}
    matrix_desc_ref::Ref{Lib.matrix_desc_t}

    function Optimizer()
        params_ref = Ref{Lib.pdhg_parameters_t}()
        Lib.set_default_parameters(Base.unsafe_convert(Ptr{Lib.pdhg_parameters_t}, params_ref))
        desc_val = Lib.matrix_desc_t(ntuple(_ -> 0x00, sizeof(Lib.matrix_desc_t)))
        desc_ref = Ref{Lib.matrix_desc_t}(desc_val)
        return new(
            nothing,
            C_NULL,
            C_NULL,
            params_ref[],
            nothing,
            false,
            false,
            Cdouble[],
            Cdouble[],
            Cdouble[],
            Cdouble[],
            Cdouble[],
            Cdouble[],
            Cint[],
            Cint[],
            Cdouble[],
            desc_ref,
        )
    end
end

function MOI.default_cache(::Optimizer, ::Type)
    return MOI.Utilities.UniversalFallback(OptimizerCache())
end

# ====================
#   Empty
# ====================

function MOI.is_empty(optimizer::Optimizer)
    return isnothing(optimizer.result) && optimizer.native_problem_ptr == C_NULL
end

function MOI.empty!(optimizer::Optimizer)
    if optimizer.native_result_ptr != C_NULL
        Lib.cupdlpx_result_free(optimizer.native_result_ptr)
        optimizer.native_result_ptr = C_NULL
    end
    if optimizer.native_problem_ptr != C_NULL
        Lib.lp_problem_free(optimizer.native_problem_ptr)
        optimizer.native_problem_ptr = C_NULL
    end
    optimizer.result = nothing
    optimizer.sets = nothing
    optimizer.max_sense = false
    empty!(optimizer.c)
    empty!(optimizer.obj_const)
    empty!(optimizer.row_lower)
    empty!(optimizer.row_upper)
    empty!(optimizer.var_lower)
    empty!(optimizer.var_upper)
    empty!(optimizer.A_colptr)
    empty!(optimizer.A_rowval)
    empty!(optimizer.A_nzval)
    return
end

# ====================
#   Helper: Immutable Update
# ====================
function _update_immutable(obj::T, field::Symbol, value) where {T}
    args = map(fieldnames(T)) do f
        if f == field
            convert(fieldtype(T, f), value)
        else
            getfield(obj, f)
        end
    end
    return T(args...)
end

# ====================
#   Parameters
# ====================

MOI.get(::Optimizer, ::MOI.SolverName) = "CuPDLPx"

function MOI.supports(::Optimizer, param::MOI.RawOptimizerAttribute)
    s = Symbol(param.name)
    return hasfield(Lib.pdhg_parameters_t, s) || 
           hasfield(Lib.termination_criteria_t, s)
end

function MOI.set(optimizer::Optimizer, param::MOI.RawOptimizerAttribute, value)
    s = Symbol(param.name)
    if hasfield(Lib.pdhg_parameters_t, s)
        optimizer.parameters = _update_immutable(optimizer.parameters, s, value)
    elseif hasfield(Lib.termination_criteria_t, s)
        current_criteria = optimizer.parameters.termination_criteria
        new_criteria = _update_immutable(current_criteria, s, value)
        optimizer.parameters = _update_immutable(optimizer.parameters, :termination_criteria, new_criteria)
    else
        throw(MOI.UnsupportedAttribute(param))
    end
    return
end

function MOI.get(optimizer::Optimizer, param::MOI.RawOptimizerAttribute)
    s = Symbol(param.name)
    if hasfield(Lib.pdhg_parameters_t, s)
        return getfield(optimizer.parameters, s)
    elseif hasfield(Lib.termination_criteria_t, s)
        return getfield(optimizer.parameters.termination_criteria, s)
    else
        throw(MOI.UnsupportedAttribute(param))
    end
end

MOI.supports(::Optimizer, ::MOI.TimeLimitSec) = true

function MOI.set(optimizer::Optimizer, ::MOI.TimeLimitSec, value::Real)
    current_criteria = optimizer.parameters.termination_criteria
    new_criteria = _update_immutable(current_criteria, :time_sec_limit, Float64(value))
    optimizer.parameters = _update_immutable(optimizer.parameters, :termination_criteria, new_criteria)
    return
end

function MOI.get(optimizer::Optimizer, ::MOI.TimeLimitSec)
    return optimizer.parameters.termination_criteria.time_sec_limit
end

MOI.supports(::Optimizer, ::MOI.Silent) = true

function MOI.set(optimizer::Optimizer, ::MOI.Silent, value::Bool)
    optimizer.silent = value
    return
end

MOI.get(optimizer::Optimizer, ::MOI.Silent) = optimizer.silent

function MOI.supports_constraint(
    ::Optimizer,
    ::Type{MOI.VariableIndex},
    ::Type{<:BOUND_SETS},
)
    return true
end

function MOI.supports_constraint(
    ::Optimizer,
    ::Type{MOI.ScalarAffineFunction{Float64}},
    ::Type{<:BOUND_SETS},
)
    return true
end

MOI.supports(::Optimizer, ::MOI.ObjectiveSense) = true

function MOI.supports(
    ::Optimizer,
    ::MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}},
)
    return true
end

# ====================
#   Copy
# ====================

# cuPDLPx returns duals and reduced costs using the classical sign convention,
# which negates them for maximization problems. MOI's convention keeps the
# signs of the equivalent negated-objective minimization problem, so undo the
# negation here.
function _negate_if_max(optimizer::Optimizer, x)
    return optimizer.max_sense ? -x : x
end

function _fill_matrix_desc!(dest::Optimizer, m::Integer, n::Integer)
    A_csc = Lib.MatrixCSC(
        length(dest.A_rowval),
        isempty(dest.A_colptr) ? C_NULL : pointer(dest.A_colptr),
        isempty(dest.A_rowval) ? C_NULL : pointer(dest.A_rowval),
        isempty(dest.A_nzval)  ? C_NULL : pointer(dest.A_nzval),
    )
    desc_ptr = Base.unsafe_convert(Ptr{Lib.matrix_desc_t}, dest.matrix_desc_ref)
    desc_ptr.m = Cint(m)
    desc_ptr.n = Cint(n)
    desc_ptr.fmt = Lib.matrix_csc
    desc_ptr.data.csc = A_csc
    return
end

function MOI.copy_to(dest::Optimizer, src::OptimizerCache)
    MOI.empty!(dest)

    dest.max_sense = MOI.get(src, MOI.ObjectiveSense()) == MOI.MAX_SENSE
    dest.sets = src.constraints.sets

    Ab = src.constraints
    A = Ab.coefficients

    dest.A_colptr = Vector{Cint}(A.colptr)
    dest.A_rowval = Vector{Cint}(A.rowval)
    dest.A_nzval  = Vector{Cdouble}(A.nzval)
    _fill_matrix_desc!(dest, A.m, A.n)

    dest.row_lower = Vector{Cdouble}(Ab.constants.lower)
    dest.row_upper = Vector{Cdouble}(Ab.constants.upper)
    dest.var_lower = Vector{Cdouble}(src.variables.lower)
    dest.var_upper = Vector{Cdouble}(src.variables.upper)

    obj = MOI.get(src, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}())
    dest.c = zeros(Cdouble, A.n)
    for term in obj.terms
        dest.c[term.variable.value] += term.coefficient
    end
    dest.obj_const = Cdouble[MOI.constant(obj)]

    matrix_desc_ptr = Base.unsafe_convert(Ptr{Lib.matrix_desc_t}, dest.matrix_desc_ref)
    obj_sense_ref = Ref(
        dest.max_sense ? Lib.OBJECTIVE_SENSE_MAXIMIZE : Lib.OBJECTIVE_SENSE_MINIMIZE,
    )

    GC.@preserve dest begin
        prob = Lib.create_lp_problem(
            pointer(dest.c),
            matrix_desc_ptr,
            pointer(dest.row_lower),
            pointer(dest.row_upper),
            pointer(dest.var_lower),
            pointer(dest.var_upper),
            pointer(dest.obj_const),
            obj_sense_ref,
        )
        @assert prob != C_NULL
        dest.native_problem_ptr = prob
    end

    dest.result = nothing
    dest.native_result_ptr = C_NULL

    return MOI.Utilities.identity_index_map(src)
end

function MOI.copy_to(
    dest::Optimizer,
    src::MOI.Utilities.UniversalFallback{OptimizerCache},
)
    index_map = MOI.copy_to(dest, src.model)

    MOI.Utilities.pass_attributes(
        dest,
        src,
        index_map,
        MOI.get(src, MOI.ListOfVariableIndices()),
    )

    MOI.Utilities.pass_attributes(
        dest,
        MOI.Utilities.ModelFilter(src) do attr
            return !(attr isa MOI.ObjectiveSense) &&
                   !(attr isa MOI.ObjectiveFunction)
        end,
        index_map,
    )

    for (F, S) in MOI.get(src, MOI.ListOfConstraintTypesPresent())
        MOI.Utilities.pass_attributes(
            dest,
            src,
            index_map,
            MOI.get(src, MOI.ListOfConstraintIndices{F,S}()),
        )
    end

    return index_map
end

# ====================
#   Optimize
# ====================

function MOI.optimize!(dest::Optimizer)
    @assert dest.native_problem_ptr != C_NULL

    solve_params = dest.parameters
    if dest.silent
        solve_params = _update_immutable(solve_params, :verbose, false)
    end
    params_ref = Ref(solve_params)
    params_ptr = Base.unsafe_convert(Ptr{Lib.pdhg_parameters_t}, params_ref)

    GC.@preserve dest params_ref begin
        result_ptr = Lib.solve_lp_problem(dest.native_problem_ptr, params_ptr)
        @assert result_ptr != C_NULL
        dest.result = unsafe_load(result_ptr)
        dest.native_result_ptr = result_ptr
    end

    return
end

function MOI.optimize!(
    dest::Optimizer,
    src::MOI.Utilities.UniversalFallback{OptimizerCache},
)
    index_map = MOI.copy_to(dest, src)
    MOI.optimize!(dest)
    return index_map, false
end

function MOI.optimize!(dest::Optimizer, src::OptimizerCache)
    index_map = MOI.copy_to(dest, src)
    MOI.optimize!(dest)
    return index_map, false
end

function MOI.optimize!(dest::Optimizer, src::MOI.ModelLike)
    cache = MOI.default_cache(dest, Float64)
    index_map = MOI.copy_to(cache, src)
    MOI.optimize!(dest, cache)
    return index_map, false
end

# ====================
#   Result Getters
# ====================

function MOI.get(optimizer::Optimizer, ::MOI.SolveTimeSec)
    return isnothing(optimizer.result) ? 0.0 : optimizer.result.cumulative_time_sec
end

function MOI.get(optimizer::Optimizer, ::MOI.RawStatusString)
    return isnothing(optimizer.result) ? "Optimize not called" : "Solver finished"
end

const _TERMINATION_STATUS_MAP = Dict(
    Lib.TERMINATION_REASON_UNSPECIFIED => MOI.OPTIMIZE_NOT_CALLED,
    Lib.TERMINATION_REASON_OPTIMAL => MOI.OPTIMAL,
    Lib.TERMINATION_REASON_PRIMAL_INFEASIBLE => MOI.INFEASIBLE,
    Lib.TERMINATION_REASON_DUAL_INFEASIBLE => MOI.DUAL_INFEASIBLE,
    Lib.TERMINATION_REASON_INFEASIBLE_OR_UNBOUNDED => MOI.INFEASIBLE_OR_UNBOUNDED,
    Lib.TERMINATION_REASON_TIME_LIMIT => MOI.TIME_LIMIT,
    Lib.TERMINATION_REASON_ITERATION_LIMIT => MOI.ITERATION_LIMIT,
    Lib.TERMINATION_REASON_FEAS_POLISH_SUCCESS => MOI.OPTIMAL,
)

function MOI.get(optimizer::Optimizer, ::MOI.TerminationStatus)
    return isnothing(optimizer.result) ? MOI.OPTIMIZE_NOT_CALLED :
           _TERMINATION_STATUS_MAP[optimizer.result.termination_reason]
end

function MOI.get(optimizer::Optimizer, attr::MOI.ObjectiveValue)
    MOI.check_result_index_bounds(optimizer, attr)
    return optimizer.result.primal_objective_value
end

function MOI.get(optimizer::Optimizer, attr::MOI.DualObjectiveValue)
    MOI.check_result_index_bounds(optimizer, attr)
    return optimizer.result.dual_objective_value
end

const _PRIMAL_STATUS_MAP = Dict(
    Lib.TERMINATION_REASON_UNSPECIFIED => MOI.NO_SOLUTION,
    Lib.TERMINATION_REASON_OPTIMAL => MOI.FEASIBLE_POINT,
    Lib.TERMINATION_REASON_PRIMAL_INFEASIBLE => MOI.NO_SOLUTION,
    Lib.TERMINATION_REASON_DUAL_INFEASIBLE => MOI.INFEASIBILITY_CERTIFICATE,
    Lib.TERMINATION_REASON_INFEASIBLE_OR_UNBOUNDED => MOI.UNKNOWN_RESULT_STATUS,
    Lib.TERMINATION_REASON_TIME_LIMIT => MOI.UNKNOWN_RESULT_STATUS,
    Lib.TERMINATION_REASON_ITERATION_LIMIT => MOI.UNKNOWN_RESULT_STATUS,
    Lib.TERMINATION_REASON_FEAS_POLISH_SUCCESS => MOI.FEASIBLE_POINT,
)

const _DUAL_STATUS_MAP = Dict(
    Lib.TERMINATION_REASON_UNSPECIFIED => MOI.NO_SOLUTION,
    Lib.TERMINATION_REASON_OPTIMAL => MOI.FEASIBLE_POINT,
    Lib.TERMINATION_REASON_PRIMAL_INFEASIBLE => MOI.INFEASIBILITY_CERTIFICATE,
    Lib.TERMINATION_REASON_DUAL_INFEASIBLE => MOI.NO_SOLUTION,
    Lib.TERMINATION_REASON_INFEASIBLE_OR_UNBOUNDED => MOI.UNKNOWN_RESULT_STATUS,
    Lib.TERMINATION_REASON_TIME_LIMIT => MOI.UNKNOWN_RESULT_STATUS,
    Lib.TERMINATION_REASON_ITERATION_LIMIT => MOI.UNKNOWN_RESULT_STATUS,
    Lib.TERMINATION_REASON_FEAS_POLISH_SUCCESS => MOI.FEASIBLE_POINT,
)

function MOI.get(optimizer::Optimizer, attr::MOI.PrimalStatus)
    if attr.result_index > MOI.get(optimizer, MOI.ResultCount())
        return MOI.NO_SOLUTION
    end
    return _PRIMAL_STATUS_MAP[optimizer.result.termination_reason]
end

function MOI.get(optimizer::Optimizer, attr::MOI.VariablePrimal, vi::MOI.VariableIndex)
    MOI.check_result_index_bounds(optimizer, attr)
    return unsafe_load(optimizer.result.primal_solution, vi.value)
end

function MOI.get(optimizer::Optimizer, attr::MOI.DualStatus)
    if attr.result_index > MOI.get(optimizer, MOI.ResultCount())
        return MOI.NO_SOLUTION
    end
    return _DUAL_STATUS_MAP[optimizer.result.termination_reason]
end

function MOI.get(
    optimizer::Optimizer,
    attr::MOI.ConstraintDual,
    ci::MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}},
)
    MOI.check_result_index_bounds(optimizer, attr)
    row = only(MOI.Utilities.rows(optimizer.sets, ci))
    return _negate_if_max(optimizer, unsafe_load(optimizer.result.dual_solution, row))
end

function MOI.get(
    optimizer::Optimizer,
    attr::MOI.ConstraintDual,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,MOI.LessThan{Float64}},
)
    MOI.check_result_index_bounds(optimizer, attr)
    rc = _negate_if_max(optimizer, unsafe_load(optimizer.result.reduced_cost, ci.value))
    return min(0.0, rc)
end

function MOI.get(
    optimizer::Optimizer,
    attr::MOI.ConstraintDual,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,MOI.GreaterThan{Float64}},
)
    MOI.check_result_index_bounds(optimizer, attr)
    rc = _negate_if_max(optimizer, unsafe_load(optimizer.result.reduced_cost, ci.value))
    return max(0.0, rc)
end

function MOI.get(
    optimizer::Optimizer,
    attr::MOI.ConstraintDual,
    ci::MOI.ConstraintIndex{
        MOI.VariableIndex,
        <:Union{MOI.Interval{Float64},MOI.EqualTo{Float64}},
    },
)
    MOI.check_result_index_bounds(optimizer, attr)
    return _negate_if_max(optimizer, unsafe_load(optimizer.result.reduced_cost, ci.value))
end

function MOI.get(optimizer::Optimizer, ::MOI.ResultCount)
    return isnothing(optimizer.result) ? 0 : 1
end
