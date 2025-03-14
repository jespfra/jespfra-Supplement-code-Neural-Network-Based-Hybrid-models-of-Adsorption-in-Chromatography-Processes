

# LRM model used for test case 

function model(; dataset, sol_times)
    """
        A function to create a model for the LRM model. 
        Dataset corresponds to the dataset used for the model. 
        Hence, the parameters are set accordingly.
        input: dataset, sol_times
        dataset 1 is the AcidPro_ProPro dataset
        dataset 2 is the H2O_AcidPro dataset
        dataset 3 is the Poh_ProPro dataset
        dataset 3a is the reactive dataset fig3a
        dataset 3b is the reactive dataset fig3b
        dataset 4a is the reactive dataset fig4a
        dataset 4b is the reactive dataset fig4b

        sol_times is the solution times for the dataset
    """

    nComp = 4
    model = OrderedDict(
        "root" => OrderedDict(
            "input" => OrderedDict(
                "model" => OrderedDict()
            )
        )
    )

    # Components = POH, AcidPro (ProAcid), H2O, ProPro

    # Set elements sequentially for unit_000
    model["root"]["input"]["model"]["unit_000"] = OrderedDict()
    model["root"]["input"]["model"]["unit_000"]["unit_type"] = "INLET"
    model["root"]["input"]["model"]["unit_000"]["ncomp"] = nComp
    model["root"]["input"]["model"]["unit_000"]["inlet_type"] = "PIECEWISE_CUBIC_POLY"

    model["root"]["input"]["model"]["unit_000"]["sec_000"] = OrderedDict()

    if dataset == "1"
        # The AcidPro_ProPro system 
        model["root"]["input"]["model"]["unit_000"]["sec_000"] = OrderedDict()
        model["root"]["input"]["model"]["unit_000"]["sec_000"]["const_coeff"] = [0, 10.78, 0, 1.36]

        model["root"]["input"]["model"]["unit_000"]["sec_001"] = OrderedDict()
        model["root"]["input"]["model"]["unit_000"]["sec_001"]["const_coeff"] = [0, 7.43, 0, 3.26]

        model["root"]["input"]["model"]["unit_000"]["sec_002"] = OrderedDict()
        model["root"]["input"]["model"]["unit_000"]["sec_002"]["const_coeff"] = [0, 4.56, 0, 4.90]

        model["root"]["input"]["model"]["unit_000"]["sec_003"] = OrderedDict()
        model["root"]["input"]["model"]["unit_000"]["sec_003"]["const_coeff"] = [0, 2.40, 0, 6.06]
        nsec = 5
        section_times = [0.0, 35.4, 71.7, 111.2, 139.3]
        Q = 7.88e-3 #Qf in dm3/min
        Pe = 166 # Pe = L*u/d = L*(Q/A/eps)/d
        L = 1.15 # From article
        cross_section_area = 4.15e-2 # From article
        porosity = 0.4 # From article
        Dax = Q*L/Pe/(cross_section_area * porosity) #dm2/min

    elseif dataset == "2"
        # The H2O_AcidPro system
        model["root"]["input"]["model"]["unit_000"]["sec_000"] = OrderedDict()
        model["root"]["input"]["model"]["unit_000"]["sec_000"]["const_coeff"] = [0, 4.83, 35.3, 0]

        model["root"]["input"]["model"]["unit_000"]["sec_001"] = OrderedDict()
        model["root"]["input"]["model"]["unit_000"]["sec_001"]["const_coeff"] = [0, 9.10, 17.51, 0]

        model["root"]["input"]["model"]["unit_000"]["sec_002"] = OrderedDict()
        model["root"]["input"]["model"]["unit_000"]["sec_002"]["const_coeff"] = [0, 10.88, 9.56, 0]

        model["root"]["input"]["model"]["unit_000"]["sec_003"] = OrderedDict()
        model["root"]["input"]["model"]["unit_000"]["sec_003"]["const_coeff"] = [0, 12.1, 4.60, 0]

        model["root"]["input"]["model"]["unit_000"]["sec_004"] = OrderedDict()
        model["root"]["input"]["model"]["unit_000"]["sec_004"]["const_coeff"] = [0, 13.0, 1.42, 0]

        model["root"]["input"]["model"]["unit_000"]["sec_005"] = OrderedDict()
        model["root"]["input"]["model"]["unit_000"]["sec_005"]["const_coeff"] = [0, 0.47, 54.2, 0]
        nsec = 7
        section_times = [0.0, 31, 61.9, 93, 125, 155.7, 179.2]

        Q = 7.88e-3 #Qf in dm3/min
        Pe = 166 # Pe = L*u/d = L*(Q/A/eps)/d
        # Q = 4.96e-3 #Qf in dm3/min
        # Pe = 181 # Pe = L*u/d = L*(Q/A/eps)/d
        L = 1.15 # From article
        cross_section_area = 4.15e-2 # From article
        porosity = 0.4 # From article
        Dax = Q*L/Pe/(cross_section_area * porosity) #dm2/min
       
    elseif dataset == "3"
        # The Poh_ProPro system
        model["root"]["input"]["model"]["unit_000"]["sec_000"] = OrderedDict()
        model["root"]["input"]["model"]["unit_000"]["sec_000"]["const_coeff"] = [11.64, 0, 0, 0.95]

        model["root"]["input"]["model"]["unit_000"]["sec_001"] = OrderedDict()
        model["root"]["input"]["model"]["unit_000"]["sec_001"]["const_coeff"] = [8.8, 0, 0, 2.53]

        model["root"]["input"]["model"]["unit_000"]["sec_002"] = OrderedDict()
        model["root"]["input"]["model"]["unit_000"]["sec_002"]["const_coeff"] = [6.23, 0, 0, 3.95]

        model["root"]["input"]["model"]["unit_000"]["sec_003"] = OrderedDict()
        model["root"]["input"]["model"]["unit_000"]["sec_003"]["const_coeff"] = [3.67, 0, 0, 5.38]

        model["root"]["input"]["model"]["unit_000"]["sec_004"] = OrderedDict()
        model["root"]["input"]["model"]["unit_000"]["sec_004"]["const_coeff"] = [1.33, 0, 0, 6.67]

        model["root"]["input"]["model"]["unit_000"]["sec_005"] = OrderedDict()
        model["root"]["input"]["model"]["unit_000"]["sec_005"]["const_coeff"] = [13.23, 0, 0, 0.07]
        nsec = 7
        section_times = [0.0, 27.287, 57.387, 86.267, 116.667, 147.067, 174.427]

        Q = 7.88e-3 #Qf in dm3/min
        Pe = 166 # Pe = L*u/d = L*(Q/A/eps)/d
        L = 1.15 # From article
        cross_section_area = 4.15e-2 # From article
        porosity = 0.4 # From article
        Dax = Q*L/Pe/(cross_section_area * porosity) #dm2/min

    elseif dataset == "fig3a"
        model["root"]["input"]["model"]["unit_000"]["sec_000"]["const_coeff"] = [0.0, 13.36, 0, 0] # mol/L
        Q = 2e-3 # Qf in L/min
        L = 1.21 # dm, from thesis
        cross_section_area = 0.26/4^2*pi # dm^2 from thesis
        Dax = 3e-3
        porosity = 0.42 # from thesis 
        nsec = 1
        section_times = [0.0, sol_times[end]]
    elseif dataset == "fig3b"
        model["root"]["input"]["model"]["unit_000"]["sec_000"]["const_coeff"] = [13.2, 0.0, 0, 0] # mol/L
        Q = 1.2e-3 # Qf in L/min
        L = 1.21 # dm, from thesis
        cross_section_area = 0.26/4^2*pi # dm^2 from thesis
        Dax = 3e-3
        porosity = 0.42 # from thesis 
        nsec = 1
        section_times = [0.0, sol_times[end]]
    elseif dataset == "fig4a"
        model["root"]["input"]["model"]["unit_000"]["sec_000"]["const_coeff"] = [7.29+0.75, 4.6+0.75, 0, 0] # mol/L
        Q = 1.8e-3 # Qf in L/min
        L = 1.21 # dm, from thesis
        cross_section_area = 0.26/4^2*pi # dm^2 from thesis
        Dax = 3e-3
        porosity = 0.42 # from thesis 
        nsec = 1
        section_times = [0.0, sol_times[end]]
    elseif dataset == "fig4b"
        model["root"]["input"]["model"]["unit_000"]["sec_000"]["const_coeff"] = [9.63+0.612, 2.65+0.612, 0, 0] # mol/L
        Q = 1.8e-3 # Qf in L/min
        L = 1.21 # dm, from thesis
        cross_section_area = 0.26/4^2*pi # dm^2 from thesis
        Dax = 3e-3
        porosity = 0.42 # from thesis 
        nsec = 1
        section_times = [0.0, sol_times[end]]
    end

    # If running on training data, the section times should be the same as the solution times and the number of sections should be reduced by one. 
    if sol_times[end] < section_times[end]
        nsec -= 1
        section_times = section_times[1:end-1]
    end


    # Set elements sequentially for unit_001
    model["root"]["input"]["model"]["unit_001"] = OrderedDict()
    model["root"]["input"]["model"]["unit_001"]["unit_type"] = "LUMPED_RATE_MODEL_WITHOUT_PORES"
    model["root"]["input"]["model"]["unit_001"]["ncomp"] = nComp
    model["root"]["input"]["model"]["unit_001"]["col_porosity"] = porosity
    model["root"]["input"]["model"]["unit_001"]["col_length"] = L
    model["root"]["input"]["model"]["unit_001"]["cross_section_area"] = cross_section_area # dm^2 from thesis
    model["root"]["input"]["model"]["unit_001"]["col_dispersion"] = Dax #? determine from correlations 
    model["root"]["input"]["model"]["unit_001"]["adsorption_model"] = "MULTI_COMPONENT_LANGMUIR_LDF"

    model["root"]["input"]["model"]["unit_001"]["adsorption"] = OrderedDict()
    model["root"]["input"]["model"]["unit_001"]["adsorption"]["is_kinetic"] = true
    model["root"]["input"]["model"]["unit_001"]["adsorption"]["MCL_LDF_QMAX"] = [9.13, 10.06, 43.07, 5.11] # mol/L
    model["root"]["input"]["model"]["unit_001"]["adsorption"]["MCL_LDF_Keq"] = [11.66, 9.04, 2.35, 5.08] # L/mol
    model["root"]["input"]["model"]["unit_001"]["adsorption"]["MCL_LDF_kL"] = 0.25


    # Set initial conditions for the dataset
    # Components = POH, AcidPro (ProAcid), H2O, ProPro
    if dataset == "1"
        model["root"]["input"]["model"]["unit_001"]["init_c"] = [0.0, 13.2, 0.0, 0.0] # mol/L
        model["root"]["input"]["model"]["unit_001"]["init_q"] = [0.0, 10.06*9.04*13.2/(1+13.2*9.04), 0, 0] # mol/L
    elseif dataset == "2"
        model["root"]["input"]["model"]["unit_001"]["init_c"] = [0.0, 0, 55.1, 0] # mol/L
        model["root"]["input"]["model"]["unit_001"]["init_q"] = [0.0, 0, 43.07*2.35*55.1/(1+2.35*55.1), 0] # mol/L
    elseif dataset == "3"
        model["root"]["input"]["model"]["unit_001"]["init_c"] = [13.23, 0.0, 0.0, 0.0] # mol/L
        model["root"]["input"]["model"]["unit_001"]["init_q"] = [9.13*11.66*13.23/(1+11.66*13.23), 0, 0, 0] # mol/L
    elseif dataset == "fig3a"
        model["root"]["input"]["model"]["unit_001"]["init_c"] = [13.36, 0, 0, 0] # mol/L
        model["root"]["input"]["model"]["unit_001"]["init_q"] = [9.13*11.66*13.36/(1+11.66*13.36), 0, 0, 0] # mol/L
    elseif dataset == "fig3b"
        model["root"]["input"]["model"]["unit_001"]["init_c"] = [0.0, 12.8, 0, 0] # mol/L
        model["root"]["input"]["model"]["unit_001"]["init_q"] = [0, 10.06*9.04*12.8/(1+12.8*9.04), 0, 0] # mol/L
    elseif dataset == "fig4a"
        model["root"]["input"]["model"]["unit_001"]["init_c"] = [13.36, 0, 0, 0] # mol/L
        model["root"]["input"]["model"]["unit_001"]["init_q"] = [9.13*11.66*13.36/(1+11.66*13.36), 0, 0, 0] # mol/L
    elseif dataset == "fig4b"
        model["root"]["input"]["model"]["unit_001"]["init_c"] = [13.36, 0, 0, 0] # mol/L
        model["root"]["input"]["model"]["unit_001"]["init_q"] = [9.13*11.66*13.36/(1+11.66*13.36), 0, 0, 0] # mol/L
    end

	
	# Reaction if dataset is reactive
    if dataset in ["fig3a", "fig3b", "fig4a", "fig4b"]
        model["root"]["input"]["model"]["unit_001"]["reaction_model_solid"] = "reaction_2"
        model["root"]["input"]["model"]["unit_001"]["reaction"] = OrderedDict()
        K_eq_inf = 8.0e3 #7.504
        deltaH_eq = 1.9e4 #-4.161e3 #J/mol
        k1_0 = 5.8e5*60 #6.848e7*60 #mol/min/eq
        deltaH_k1 = 5.3e4 #5.918e4 #J/mol 
        T = 313. # K for experiment 1, table 3
        k_eq_correction = 1#3
        k1_correction = 1#2
        k1 = k1_0*exp(-deltaH_k1/8.314/T)/k1_correction
        Keq = K_eq_inf*exp(-deltaH_eq/8.314/T)/k_eq_correction
        density = 0.3
        
        Vm = [0.8/60.10, 0.99/74.08, 0.876/116.16, 1/18.02] #Relative to H20 density, from Pubchem
        # Thermodynamic model 
        # Clapeyron definition: 
        # G_ij = exp(-c*tau) - c corresponds to alpha in article for Clapeyron 
        # tau_ij = a_ij + b_ij/T - corresponds to a_ij=0 and b_ij=delta_g_ij/R
        # Therefore:
        # a is zero matrix as there are no constant i.e., no dependence on T 
        # b corresponds to the delta_g terms, note g_ij is different from g_ji. 
        # Tmodel = NRTL(["1-propanol","propanoic acid", "propyl-propionate", "water"], puremodel =BasicIdeal,
        # 		userlocations = (a = [0.0 0 0 0; 0 0 0 0; 0 0 0 0; 0 0 0 0],
        # 						b = [0.0 1754 566 1744.7; 1120.8 0.0 590 1631.9; 236.9 32.1 0.0 3505; -93.5 -1823 1142.5 0.0],
        # 						c = [0.0 0.3 0.3 0.3; 0.3 0.0 0.3 0.3; 0.3 0.3 0.0 0.3; 0.3 0.3 0.3 0.0], 
        #                         )
        # 					)

        # Thermodynamic NRTL model in the order of: "1-propanol","propanoic acid", "water", "propyl-propionate", 
        a = [0.0 0 0 0; 0 0 0 0; 0 0 0 0; 0 0 0 0]
        b = [0.0 674.9 704 391.7; -712 0.0 672.6 384.5; 766.6 4244 0.0 3302.4; -677.4 224.6 2237.8 0.0] ./ 8.314
        c = [0.0 0.3 0.3 0.3; 0.3 0.0 0.3 0.3; 0.3 0.3 0.0 0.3; 0.3 0.3 0.3 0.0]
        Tmodel = (a,b,c)
        

        # b = [0.0 1754 566 1744.7; 1120.8 0.0 590 1631.9; 236.9 32.1 0.0 3505; -93.5 -1823 1142.5 0.0]
        #b = [0.0 1120.8 236.9 -93.5; 1754 0.0 32.1 -1823; 566 590 0.0 1142.5; 1744.7 1631.9 3205 0.0] ./ 8.314                        
        # activity_coefficient(Tmodel, 1e5, T, [0.5, 0, 0, 0.5])

        model["root"]["input"]["model"]["unit_001"]["reaction"]["k1"] = k1
        model["root"]["input"]["model"]["unit_001"]["reaction"]["Keq"] = Keq
        model["root"]["input"]["model"]["unit_001"]["reaction"]["K"] = [11.66, 9.04, 2.35, 5.08]
        model["root"]["input"]["model"]["unit_001"]["reaction"]["stoich"]= [-1., -1, 1, 1]
        model["root"]["input"]["model"]["unit_001"]["reaction"]["Vm"] = Vm
        model["root"]["input"]["model"]["unit_001"]["reaction"]["Tmodel"]= Tmodel
        model["root"]["input"]["model"]["unit_001"]["reaction"]["density"]= density
        model["root"]["input"]["model"]["unit_001"]["reaction"]["T"]= T
    end
	

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
    model["root"]["input"]["solver"]["sections"]["nsec"] = nsec
    model["root"]["input"]["solver"]["sections"]["section_times"] = section_times # min


    # Set elements for connections
    model["root"]["input"]["model"]["connections"] = OrderedDict()
    model["root"]["input"]["model"]["connections"]["nswitches"] = 1
    model["root"]["input"]["model"]["connections"]["switch_000"] = OrderedDict()
    model["root"]["input"]["model"]["connections"]["switch_000"]["section"] = 0
    model["root"]["input"]["model"]["connections"]["switch_000"]["connections"] = [0, 1, -1, -1, Q, 
                                                                                1, 2, -1, -1, Q]


    # Set elements for user_solution_times
    model["root"]["input"]["solver"]["user_solution_times"] = sol_times #solution times depending on dataset 

    # Set elements for time_integrator
    model["root"]["input"]["solver"]["time_integrator"] = OrderedDict()
    model["root"]["input"]["solver"]["time_integrator"]["abstol"] = 5e-7
    model["root"]["input"]["solver"]["time_integrator"]["algtol"] = 5e-7
    model["root"]["input"]["solver"]["time_integrator"]["reltol"] = 5e-7



    inlets, outlets, columns, switches, solverOptions = create_units(model)

    return inlets, outlets, columns, switches, solverOptions
end


