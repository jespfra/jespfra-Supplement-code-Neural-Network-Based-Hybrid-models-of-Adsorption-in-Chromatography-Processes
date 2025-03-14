function get_initial_conditions()
    """
        A function to get initial conditions and models for the training cases. 
    """

    inlets, outlets, columns, switches, solverOptions1 = model(dataset = "1", sol_times = data_trains[1][:,end])
    rhs_test1 = hybrid_model(columns, columns[1].RHS_q, columns[1].cpp, columns[1].qq, 1, solverOptions1.nColumns, solverOptions1.idx_units, switches, solverOptions1, c_scale_data[2], 1.0)
    inlets, outlets, columns, switches, solverOptions2 = model(dataset = "2", sol_times = data_trains[2][:,end])
    rhs_test2 = hybrid_model(columns, columns[1].RHS_q, columns[1].cpp, columns[1].qq, 1, solverOptions2.nColumns, solverOptions2.idx_units, switches, solverOptions2, c_scale_data[2], 1.0)
    inlets, outlets, columns, switches, solverOptions3 = model(dataset = "3", sol_times = data_trains[3][:,end])
    rhs_test3 = hybrid_model(columns, columns[1].RHS_q, columns[1].cpp, columns[1].qq, 1, solverOptions3.nColumns, solverOptions3.idx_units, switches, solverOptions3, c_scale_data[2], 1.0)
    rhs_tests = (rhs_test1, rhs_test2, rhs_test3)	
    x0 = [solverOptions1.x0 solverOptions2.x0 solverOptions3.x0] 

    return rhs_tests, x0
end
