using Test
using CuPDLPx

const Lib = CuPDLPx.LibCuPDLPx

@testset "LibCuPDLPx Translation Tests" begin

    # ==========================================
    # 1. Basic Enum Mapping Tests
    # ==========================================
    @testset "Enum Mapping" begin
        @test Int(Lib.TERMINATION_REASON_OPTIMAL) == 1
        @test Int(Lib.TERMINATION_REASON_TIME_LIMIT) == 5
        
        @test Int(Lib.matrix_dense) == 0
        @test Int(Lib.matrix_csr) == 1
        @test Int(Lib.matrix_csc) == 2
    end

    # ==========================================
    # 2. Union Memory Layout Tests
    # ==========================================
    @testset "Union Memory Layout (MatrixData)" begin
        @test sizeof(Lib.MatrixData) == 32
        data_blob = Lib.MatrixData(ntuple(_ -> UInt8(0), 32))
        data_ref = Ref(data_blob)
        data_ptr = Base.unsafe_convert(Ptr{Lib.MatrixData}, data_ref)
        
        @test data_ptr.csr isa Ptr{Lib.MatrixCSR}
        @test Int(data_ptr) == Int(data_ptr.csr)
        println("   > MatrixData Union logic verified.")
    end

    # ==========================================
    # 3. C Function Call Test
    # ==========================================
    @testset "C Function Call: set_default_parameters" begin
        params_ref = Ref{Lib.pdhg_parameters_t}()
        params_ptr = Base.unsafe_convert(Ptr{Lib.pdhg_parameters_t}, params_ref)
        Lib.set_default_parameters(params_ptr)
        params = params_ref[]
        
        @test params.verbose isa Integer
        @test params.termination_criteria.time_sec_limit > 0
        println("   > Successfully called set_default_parameters.")
    end

    # ==========================================
    # 4. Struct Size Check
    # ==========================================
    @testset "Struct Size Sanity Check" begin
        @test sizeof(Lib.matrix_desc_t) == 48
    end

    # ==========================================
    # 5. Integration Test: MPS File
    # ==========================================
    @testset "MPS File Solve (AFIRO)" begin
        mps_content = """
            NAME          AFIRO
            ROWS
            E  R09
            E  R10
            L  X05
            L  X21
            E  R12
            E  R13
            L  X17
            L  X18
            L  X19
            L  X20
            E  R19
            E  R20
            L  X27
            L  X44
            E  R22
            E  R23
            L  X40
            L  X41
            L  X42
            L  X43
            L  X45
            L  X46
            L  X47
            L  X48
            L  X49
            L  X50
            L  X51
            N  COST
            COLUMNS
                X01       X48               .301   R09                -1.
                X01       R10              -1.06   X05                 1.
                X02       X21                -1.   R09                 1.
                X02       COST               -.4
                X03       X46                -1.   R09                 1.
                X04       X50                 1.   R10                 1.
                X06       X49               .301   R12                -1.
                X06       R13              -1.06   X17                 1.
                X07       X49               .313   R12                -1.
                X07       R13              -1.06   X18                 1.
                X08       X49               .313   R12                -1.
                X08       R13               -.96   X19                 1.
                X09       X49               .326   R12                -1.
                X09       R13               -.86   X20                 1.
                X10       X45              2.364   X17                -1.
                X11       X45              2.386   X18                -1.
                X12       X45              2.408   X19                -1.
                X13       X45              2.429   X20                -1.
                X14       X21                1.4   R12                 1.
                X14       COST              -.32
                X15       X47                -1.   R12                 1.
                X16       X51                 1.   R13                 1.
                X22       X46               .109   R19                -1.
                X22       R20               -.43   X27                 1.
                X23       X44                -1.   R19                 1.
                X23       COST               -.6
                X24       X48                -1.   R19                 1.
                X25       X45                -1.   R19                 1.
                X26       X50                 1.   R20                 1.
                X28       X47               .109   R22               -.43
                X28       R23                 1.   X40                 1.
                X29       X47               .108   R22               -.43
                X29       R23                 1.   X41                 1.
                X30       X47               .108   R22               -.39
                X30       R23                 1.   X42                 1.
                X31       X47               .107   R22               -.37
                X31       R23                 1.   X43                 1.
                X32       X45              2.191   X40                -1.
                X33       X45              2.219   X41                -1.
                X34       X45              2.249   X42                -1.
                X35       X45              2.279   X43                -1.
                X36       X44                1.4   R23                -1.
                X36       COST              -.48
                X37       X49                -1.   R23                 1.
                X38       X51                 1.   R22                 1.
                X39       R23                 1.   COST               10.
            RHS
                B         X50               310.   X51               300.
                B         X05                80.   X17                80.
                B         X27               500.   R23                44.
                B         X40               500.
            ENDATA
        """
        mps_path = joinpath(tempdir(), "afiro.mps")
        write(mps_path, mps_content)
        println("   > Created temp MPS file at: $mps_path")

        prob = Lib.read_mps_file(mps_path)
        @test prob != C_NULL
        
        prob_data = unsafe_load(prob)
        println("   > Loaded Problem: $(prob_data.num_variables) vars, $(prob_data.num_constraints) cons")
        @test prob_data.num_variables > 0

        params_ref = Ref{Lib.pdhg_parameters_t}()
        Lib.set_default_parameters(Base.unsafe_convert(Ptr{Lib.pdhg_parameters_t}, params_ref))
        # Set verbose explicitly to check propagation through the native API
        params_val = params_ref[]
        params_val = Lib.pdhg_parameters_t(
            params_val.l_inf_ruiz_iterations,
            params_val.has_pock_chambolle_alpha,
            params_val.pock_chambolle_alpha,
            params_val.bound_objective_rescaling,
            true,  # verbose
            params_val.termination_evaluation_frequency,
            params_val.sv_max_iter,
            params_val.sv_tol,
            params_val.termination_criteria,
            params_val.restart_params,
            params_val.reflection_coefficient,
            params_val.feasibility_polishing,
            params_val.optimality_norm,
            params_val.presolve,
            params_val.matrix_zero_tol,
        )
        params_ref[] = params_val
        @test params_ref[].verbose == true

        params_ptr = Base.unsafe_convert(Ptr{Lib.pdhg_parameters_t}, params_ref)
        
        result_ptr = Lib.solve_lp_problem(prob, params_ptr)
        @test result_ptr != C_NULL
        
        result = unsafe_load(result_ptr)
        println("   > MPS Solve Status: $(result.termination_reason)")
        println("   > MPS Primal Obj: $(result.primal_objective_value)")

        target_obj = -464.75
        @test isapprox(result.primal_objective_value, target_obj, rtol=1e-2)

        Lib.cupdlpx_result_free(result_ptr)
        Lib.lp_problem_free(prob)
        rm(mps_path) 
    end

    # ==========================================
    # 6. Integration Test: Direct Matrix Construction
    # ==========================================
    @testset "Direct API Solve" begin
        println("\n   > Starting Direct API Solve Test...")

        # 1. Objective Vector
        c = Cdouble[-1.0, -1.0, -2.0]
        obj_const = Cdouble[0.0]

        # 2. Variable Bounds
        var_lb = Cdouble[0.0, 0.0, 0.0]
        var_ub = Cdouble[1.0, 1.0, 1.0]

        # 3. Constraint Matrix Data
        csr_vals    = Cdouble[1.0, 2.0, 3.0, -1.0, -1.0]
        csr_col_ind = Cint[0, 1, 2, 0, 1]
        csr_row_ptr = Cint[0, 3, 5]
        nnz = Cint(5)

        # 4. Constraint Bounds
        con_lb = Cdouble[-Inf, -Inf]
        con_ub = Cdouble[4.0, -1.0]

        # 5. Refs
        A_desc_ref = Ref{Lib.matrix_desc_t}()
        params_ref = Ref{Lib.pdhg_parameters_t}()

        # A. Construct MatrixCSR
        A_csr = Lib.MatrixCSR(
            nnz,
            pointer(csr_row_ptr),
            pointer(csr_col_ind),
            pointer(csr_vals)
        )

        # B. Construct matrix_desc_t        
        A_desc_ref[] = Lib.matrix_desc_t(ntuple(_ -> UInt8(0), 48)) 
        A_desc_ptr = Base.unsafe_convert(Ptr{Lib.matrix_desc_t}, A_desc_ref)

        A_desc_ptr.m = Cint(2)  # m_cons = 2
        A_desc_ptr.n = Cint(3)  # n_vars = 3

        A_desc_ptr.fmt = Lib.matrix_csr
        A_desc_ptr.data.csr = A_csr
        # Create LP Problem
        prob = Lib.create_lp_problem(
            pointer(c),
            A_desc_ptr,
            pointer(con_lb),
            pointer(con_ub),
            pointer(var_lb),
            pointer(var_ub),
            pointer(obj_const),
            Ref(Lib.OBJECTIVE_SENSE_MINIMIZE)
        )
        @test prob != C_NULL

        # Setup Parameters
        Lib.set_default_parameters(Base.unsafe_convert(Ptr{Lib.pdhg_parameters_t}, params_ref))
        params_ptr = Base.unsafe_convert(Ptr{Lib.pdhg_parameters_t}, params_ref)

        # Solve
        result_ptr = Lib.solve_lp_problem(prob, params_ptr)
        @test result_ptr != C_NULL
        
        result = unsafe_load(result_ptr)

        println("   > Direct Solve Status: $(result.termination_reason)")
        println("   > Direct Solve Obj: $(result.primal_objective_value)")
        @test result.termination_reason == Lib.TERMINATION_REASON_OPTIMAL
        @test isapprox(result.primal_objective_value, -3.0, atol=1e-4)

        Lib.cupdlpx_result_free(result_ptr)
        Lib.lp_problem_free(prob)
        println("   > Direct API Solve test passed.")
    end

end

include("MOI_wrapper.jl")
