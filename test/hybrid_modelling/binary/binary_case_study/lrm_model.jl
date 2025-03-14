

# LRM model used for test case 

function model(; sol_times = "default")
    nComp = 2
    model = OrderedDict(
        "root" => OrderedDict(
            "input" => OrderedDict(
                "model" => OrderedDict()
            )
        )
    )


    # Set elements sequentially for unit_000
    model["root"]["input"]["model"]["unit_000"] = OrderedDict()
    model["root"]["input"]["model"]["unit_000"]["unit_type"] = "INLET"
    model["root"]["input"]["model"]["unit_000"]["ncomp"] = nComp
    model["root"]["input"]["model"]["unit_000"]["inlet_type"] = "PIECEWISE_CUBIC_POLY"

    model["root"]["input"]["model"]["unit_000"]["sec_000"] = OrderedDict()
    model["root"]["input"]["model"]["unit_000"]["sec_000"]["const_coeff"] = [11.64, 0.95]

    model["root"]["input"]["model"]["unit_000"]["sec_001"] = OrderedDict()
	model["root"]["input"]["model"]["unit_000"]["sec_001"]["const_coeff"] = [8.8, 2.53]

    model["root"]["input"]["model"]["unit_000"]["sec_002"] = OrderedDict()
	model["root"]["input"]["model"]["unit_000"]["sec_002"]["const_coeff"] = [6.23, 3.95]

    model["root"]["input"]["model"]["unit_000"]["sec_003"] = OrderedDict()
	model["root"]["input"]["model"]["unit_000"]["sec_003"]["const_coeff"] = [3.67, 5.38]

    model["root"]["input"]["model"]["unit_000"]["sec_004"] = OrderedDict()
	model["root"]["input"]["model"]["unit_000"]["sec_004"]["const_coeff"] = [1.33, 6.67]

    model["root"]["input"]["model"]["unit_000"]["sec_005"] = OrderedDict()
	model["root"]["input"]["model"]["unit_000"]["sec_005"]["const_coeff"] = [13.23, 0.07]


    # Set elements sequentially for unit_001
    model["root"]["input"]["model"]["unit_001"] = OrderedDict()
    model["root"]["input"]["model"]["unit_001"]["unit_type"] = "LUMPED_RATE_MODEL_WITHOUT_PORES"
    model["root"]["input"]["model"]["unit_001"]["ncomp"] = nComp
    model["root"]["input"]["model"]["unit_001"]["col_porosity"] = 0.4
    Pe = 166 # L*u/D
    Q = 7.88e-3 #Qf in dm3/min
    L = 1.15 #dm
    model["root"]["input"]["model"]["unit_001"]["col_length"] = L
    model["root"]["input"]["model"]["unit_001"]["cross_section_area"] = 4.15e-2 # dm^2
    model["root"]["input"]["model"]["unit_001"]["col_dispersion"] =  Q/(4.15e-2*0.4)*L/Pe #dm^2/min
    model["root"]["input"]["model"]["unit_001"]["adsorption_model"] = "MULTI_COMPONENT_LANGMUIR"

    model["root"]["input"]["model"]["unit_001"]["adsorption"] = OrderedDict()
    model["root"]["input"]["model"]["unit_001"]["adsorption"]["is_kinetic"] = true
    model["root"]["input"]["model"]["unit_001"]["adsorption"]["MCL_QMAX"] = [10, 10]
    model["root"]["input"]["model"]["unit_001"]["adsorption"]["MCL_KA"] = [0.1, 0.1]
    model["root"]["input"]["model"]["unit_001"]["adsorption"]["MCL_KD"] = [1.0, 1]

    model["root"]["input"]["model"]["unit_001"]["init_c"] = [13.23, 1e-4]
    model["root"]["input"]["model"]["unit_001"]["init_q"] = [9.13*11.66*13.23/(1+11.66*13.23), 1e-4]

    model["root"]["input"]["model"]["unit_001"]["discretization"] = OrderedDict()
    model["root"]["input"]["model"]["unit_001"]["discretization"]["polyDeg"] = 4
    model["root"]["input"]["model"]["unit_001"]["discretization"]["ncol"] = 16
    model["root"]["input"]["model"]["unit_001"]["discretization"]["exact_integration"] = 1
    model["root"]["input"]["model"]["unit_001"]["discretization"]["nbound"] = ones(Bool, nComp)

    # Set elements for unit_002
    model["root"]["input"]["model"]["unit_002"] = OrderedDict()
    model["root"]["input"]["model"]["unit_002"]["unit_type"] = "OUTLET"
    model["root"]["input"]["model"]["unit_002"]["ncomp"] = nComp


    # Set elements for solver
    model["root"]["input"]["solver"] = OrderedDict("sections" => OrderedDict())
    model["root"]["input"]["solver"]["sections"]["nsec"] = 6
    model["root"]["input"]["solver"]["sections"]["section_times"] = [0.0, 27.287, 57.387, 86.267, 116.667, 147.067, 174.427] #[0.0, 30.087, 60.187, 90.067, 119.707, 147.067]
    model["root"]["input"]["solver"]["sections"]["section_continuity"] = [0]


    # Set elements for connections
    model["root"]["input"]["model"]["connections"] = OrderedDict()
    model["root"]["input"]["model"]["connections"]["nswitches"] = 1
    model["root"]["input"]["model"]["connections"]["switch_000"] = OrderedDict()
    model["root"]["input"]["model"]["connections"]["switch_000"]["section"] = 0
    model["root"]["input"]["model"]["connections"]["switch_000"]["connections"] = [0, 1, -1, -1, Q, 
                                                                                1, 2, -1, -1, Q]


    # Set elements for user_solution_times
    if sol_times == "default"
        model["root"]["input"]["solver"]["user_solution_times"] = vcat([0], collect(0.687:0.7:83.987), collect(84.747:0.76:174.427)) #[0.0, 27.287, 57.387, 86.267, 116.667, 147.067, 174.427]
    else
        model["root"]["input"]["solver"]["user_solution_times"] = sol_times
    end

    # Set elements for time_integrator
    model["root"]["input"]["solver"]["time_integrator"] = OrderedDict()
    model["root"]["input"]["solver"]["time_integrator"]["abstol"] = 5e-7
    model["root"]["input"]["solver"]["time_integrator"]["algtol"] = 5e-7
    model["root"]["input"]["solver"]["time_integrator"]["reltol"] = 5e-7



    inlets, outlets, columns, switches, solverOptions = create_units(model)

    return inlets, outlets, columns, switches, solverOptions
end


