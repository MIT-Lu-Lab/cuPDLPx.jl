module TestMOI

using Test
import MathOptInterface as MOI
import CuPDLPx

function test_runtests()
    optimizer = CuPDLPx.Optimizer()
    MOI.set(optimizer, MOI.Silent(), true) # comment this to enable output
    model = MOI.Bridges.full_bridge_optimizer(
        MOI.Utilities.CachingOptimizer(
            MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
            optimizer,
        ),
        Float64,
    )
    config = MOI.Test.Config(
        rtol = 1e-1,
        atol = 1e-1,
        exclude = Any[
            MOI.ConstraintBasisStatus,
            MOI.VariableBasisStatus,
            MOI.ConstraintName,
            MOI.VariableName,
            MOI.ObjectiveBound,
            MOI.SolverVersion,
        ],
    )
    MOI.Test.runtests(
        model,
        config,
        # failed tests
        # some implementations not supported yet, such as TimeLimitSec, etc.
        exclude = [r"^test_infeasible_MAX_SENSE$",
                   r"^test_infeasible_MAX_SENSE_offset$",
                   r"^test_infeasible_MIN_SENSE$",
                   r"^test_infeasible_MIN_SENSE_offset$",
                   r"^test_infeasible_affine_MAX_SENSE$",
                   r"^test_infeasible_affine_MAX_SENSE_offset$",
                   r"^test_infeasible_affine_MIN_SENSE$",
                   r"^test_infeasible_affine_MIN_SENSE_offset$",
                   r"^test_linear_INFEASIBLE$",
                   r"^test_linear_INFEASIBLE_2$",
                   r"^test_linear_integration_delete_variables$",
                   r"^test_solve_DualStatus_INFEASIBILITY_CERTIFICATE_EqualTo_lower$",
                   r"^test_solve_DualStatus_INFEASIBILITY_CERTIFICATE_EqualTo_upper$",
                   r"^test_solve_DualStatus_INFEASIBILITY_CERTIFICATE_GreaterThan$",
                   r"^test_solve_DualStatus_INFEASIBILITY_CERTIFICATE_Interval_lower$",
                   r"^test_solve_DualStatus_INFEASIBILITY_CERTIFICATE_Interval_upper$",
                   r"^test_solve_DualStatus_INFEASIBILITY_CERTIFICATE_LessThan$",
                   r"^test_solve_DualStatus_INFEASIBILITY_CERTIFICATE_VariableIndex_LessThan$",
                   r"^test_solve_DualStatus_INFEASIBILITY_CERTIFICATE_VariableIndex_LessThan_max$",
                  ],
        verbose = true,
    )
    return
end

function test_maximize_duals()
    # max 2x + y  s.t.  x + y <= 1,  x >= 0,  y >= 0
    # Optimal solution (1, 0) with objective value 2.
    # MOI's dual convention for MAX_SENSE keeps the signs of the equivalent
    # negated-objective minimization problem:
    #   dual of x + y <= 1 is -2 (LessThan duals are nonpositive)
    #   dual of y >= 0 is 1 (GreaterThan duals are nonnegative)
    optimizer = CuPDLPx.Optimizer()
    MOI.set(optimizer, MOI.Silent(), true)
    model = MOI.Bridges.full_bridge_optimizer(
        MOI.Utilities.CachingOptimizer(
            MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
            optimizer,
        ),
        Float64,
    )
    x = MOI.add_variable(model)
    y = MOI.add_variable(model)
    MOI.add_constraint(model, x, MOI.GreaterThan(0.0))
    cy = MOI.add_constraint(model, y, MOI.GreaterThan(0.0))
    f = MOI.ScalarAffineFunction(
        [MOI.ScalarAffineTerm(1.0, x), MOI.ScalarAffineTerm(1.0, y)],
        0.0,
    )
    c = MOI.add_constraint(model, f, MOI.LessThan(1.0))
    obj = MOI.ScalarAffineFunction(
        [MOI.ScalarAffineTerm(2.0, x), MOI.ScalarAffineTerm(1.0, y)],
        0.0,
    )
    MOI.set(model, MOI.ObjectiveFunction{typeof(obj)}(), obj)
    MOI.set(model, MOI.ObjectiveSense(), MOI.MAX_SENSE)
    MOI.optimize!(model)
    @test MOI.get(model, MOI.TerminationStatus()) == MOI.OPTIMAL
    @test isapprox(MOI.get(model, MOI.ObjectiveValue()), 2.0; atol = 1e-3)
    @test isapprox(MOI.get(model, MOI.DualObjectiveValue()), 2.0; atol = 1e-3)
    @test isapprox(MOI.get(model, MOI.VariablePrimal(), x), 1.0; atol = 1e-3)
    @test isapprox(MOI.get(model, MOI.ConstraintDual(), c), -2.0; atol = 1e-3)
    @test isapprox(MOI.get(model, MOI.ConstraintDual(), cy), 1.0; atol = 1e-3)
    return
end

function runtests()
    for name in names(@__MODULE__; all = true)
        if startswith("$(name)", "test_")
            @testset "$(name)" begin
                getfield(@__MODULE__, name)()
            end
        end
    end
    return
end

end  # module

TestMOI.runtests()
