module LibCuPDLPx

using cuPDLPx_jll
export cuPDLPx_jll

@enum termination_reason_t::UInt32 begin
    TERMINATION_REASON_UNSPECIFIED = 0
    TERMINATION_REASON_OPTIMAL = 1
    TERMINATION_REASON_PRIMAL_INFEASIBLE = 2
    TERMINATION_REASON_DUAL_INFEASIBLE = 3
    TERMINATION_REASON_INFEASIBLE_OR_UNBOUNDED = 4
    TERMINATION_REASON_TIME_LIMIT = 5
    TERMINATION_REASON_ITERATION_LIMIT = 6
    TERMINATION_REASON_FEAS_POLISH_SUCCESS = 7
end

@enum norm_type_t::UInt32 begin
    NORM_TYPE_L2 = 0
    NORM_TYPE_L_INF = 1
end

@enum objective_sense_t::UInt32 begin
    OBJECTIVE_SENSE_MINIMIZE = 0
    OBJECTIVE_SENSE_MAXIMIZE = 1
end

struct lp_problem_t
    num_variables::Cint
    num_constraints::Cint
    variable_lower_bound::Ptr{Cdouble}
    variable_upper_bound::Ptr{Cdouble}
    objective_vector::Ptr{Cdouble}
    objective_constant::Cdouble
    objective_sense::objective_sense_t
    constraint_matrix_row_pointers::Ptr{Cint}
    constraint_matrix_col_indices::Ptr{Cint}
    constraint_matrix_values::Ptr{Cdouble}
    constraint_matrix_num_nonzeros::Cint
    constraint_lower_bound::Ptr{Cdouble}
    constraint_upper_bound::Ptr{Cdouble}
    primal_start::Ptr{Cdouble}
    dual_start::Ptr{Cdouble}
end

struct restart_parameters_t
    artificial_restart_threshold::Cdouble
    sufficient_reduction_for_restart::Cdouble
    necessary_reduction_for_restart::Cdouble
    k_p::Cdouble
    k_i::Cdouble
    k_d::Cdouble
    i_smooth::Cdouble
end

struct termination_criteria_t
    eps_optimal_relative::Cdouble
    eps_feasible_relative::Cdouble
    eps_feas_polish_relative::Cdouble
    time_sec_limit::Cdouble
    iteration_limit::Cint
end

struct pdhg_parameters_t
    l_inf_ruiz_iterations::Cint
    has_pock_chambolle_alpha::Bool
    pock_chambolle_alpha::Cdouble
    bound_objective_rescaling::Bool
    verbose::Bool
    termination_evaluation_frequency::Cint
    sv_max_iter::Cint
    sv_tol::Cdouble
    termination_criteria::termination_criteria_t
    restart_params::restart_parameters_t
    reflection_coefficient::Cdouble
    feasibility_polishing::Bool
    optimality_norm::norm_type_t
    presolve::Bool
    matrix_zero_tol::Cdouble
end

struct cupdlpx_result_t
    num_variables::Cint
    num_constraints::Cint
    num_nonzeros::Cint
    num_reduced_variables::Cint
    num_reduced_constraints::Cint
    num_reduced_nonzeros::Cint
    primal_solution::Ptr{Cdouble}
    dual_solution::Ptr{Cdouble}
    reduced_cost::Ptr{Cdouble}
    total_count::Cint
    rescaling_time_sec::Cdouble
    cumulative_time_sec::Cdouble
    presolve_time::Cdouble
    presolve_status::Cint
    absolute_primal_residual::Cdouble
    relative_primal_residual::Cdouble
    absolute_dual_residual::Cdouble
    relative_dual_residual::Cdouble
    primal_objective_value::Cdouble
    dual_objective_value::Cdouble
    objective_gap::Cdouble
    relative_objective_gap::Cdouble
    max_primal_ray_infeasibility::Cdouble
    max_dual_ray_infeasibility::Cdouble
    primal_ray_linear_objective::Cdouble
    dual_ray_objective::Cdouble
    termination_reason::termination_reason_t
    feasibility_polishing_time::Cdouble
    feasibility_iteration::Cint
end

@enum matrix_format_t::UInt32 begin
    matrix_dense = 0
    matrix_csr = 1
    matrix_csc = 2
    matrix_coo = 3
end

struct MatrixData
    data::NTuple{32, UInt8}
end

function Base.getproperty(x::Ptr{MatrixData}, f::Symbol)
    f === :dense && return Ptr{MatrixDense}(x + 0)
    f === :csr && return Ptr{MatrixCSR}(x + 0)
    f === :csc && return Ptr{MatrixCSC}(x + 0)
    f === :coo && return Ptr{MatrixCOO}(x + 0)
    return getfield(x, f)
end

function Base.getproperty(x::MatrixData, f::Symbol)
    r = Ref{MatrixData}(x)
    ptr = Base.unsafe_convert(Ptr{MatrixData}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{MatrixData}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

function Base.propertynames(x::MatrixData, private::Bool = false)
    (:dense, :csr, :csc, :coo, if private
            fieldnames(typeof(x))
        else
            ()
        end...)
end

struct matrix_desc_t
    data::NTuple{48, UInt8}
end

function Base.getproperty(x::Ptr{matrix_desc_t}, f::Symbol)
    f === :m && return Ptr{Cint}(x + 0)
    f === :n && return Ptr{Cint}(x + 4)
    f === :fmt && return Ptr{matrix_format_t}(x + 8)
    f === :data && return Ptr{MatrixData}(x + 16)
    return getfield(x, f)
end

function Base.getproperty(x::matrix_desc_t, f::Symbol)
    r = Ref{matrix_desc_t}(x)
    ptr = Base.unsafe_convert(Ptr{matrix_desc_t}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{matrix_desc_t}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

function Base.propertynames(x::matrix_desc_t, private::Bool = false)
    (:m, :n, :fmt, :data, if private
            fieldnames(typeof(x))
        else
            ()
        end...)
end

function create_lp_problem(objective_c, A_desc, con_lb, con_ub, var_lb, var_ub, objective_constant, objective_sense)
    ccall((:create_lp_problem, libcupdlpx), Ptr{lp_problem_t}, (Ptr{Cdouble}, Ptr{matrix_desc_t}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{objective_sense_t}), objective_c, A_desc, con_lb, con_ub, var_lb, var_ub, objective_constant, objective_sense)
end

function set_start_values(prob, primal, dual)
    ccall((:set_start_values, libcupdlpx), Cvoid, (Ptr{lp_problem_t}, Ptr{Cdouble}, Ptr{Cdouble}), prob, primal, dual)
end

function solve_lp_problem(prob, params)
    ccall((:solve_lp_problem, libcupdlpx), Ptr{cupdlpx_result_t}, (Ptr{lp_problem_t}, Ptr{pdhg_parameters_t}), prob, params)
end

function set_default_parameters(params)
    ccall((:set_default_parameters, libcupdlpx), Cvoid, (Ptr{pdhg_parameters_t},), params)
end

function cupdlpx_result_free(results)
    ccall((:cupdlpx_result_free, libcupdlpx), Cvoid, (Ptr{cupdlpx_result_t},), results)
end

function lp_problem_free(prob)
    ccall((:lp_problem_free, libcupdlpx), Cvoid, (Ptr{lp_problem_t},), prob)
end

function read_mps_file(filename)
    ccall((:read_mps_file, libcupdlpx), Ptr{lp_problem_t}, (Ptr{Cchar},), filename)
end

struct MatrixDense
    A::Ptr{Cdouble}
end

struct MatrixCSR
    nnz::Cint
    row_ptr::Ptr{Cint}
    col_ind::Ptr{Cint}
    vals::Ptr{Cdouble}
end

struct MatrixCSC
    nnz::Cint
    col_ptr::Ptr{Cint}
    row_ind::Ptr{Cint}
    vals::Ptr{Cdouble}
end

struct MatrixCOO
    nnz::Cint
    row_ind::Ptr{Cint}
    col_ind::Ptr{Cint}
    vals::Ptr{Cdouble}
end

end # module
