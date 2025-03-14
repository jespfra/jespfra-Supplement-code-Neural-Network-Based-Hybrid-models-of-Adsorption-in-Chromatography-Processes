

# Add the include file custom to load packages and scripts. 
# the file is located on the main from which the file takes care of the rest. 
include(joinpath(@__DIR__,"..","..","..","..","..","include.jl"))
include(joinpath(@__DIR__,"..","..","..","include_hybrid.jl"))


############## Solve using a hybrid modelling approach ##############
include(joinpath(@__DIR__,"..", "..", "loss_functions.jl"))
include(joinpath(@__DIR__, "hybrid_model_setup.jl"))
include(joinpath(@__DIR__,"..", "..", "lrm_models.jl"))
include(joinpath(@__DIR__,"..", "..", "get_data.jl"))
include(joinpath(@__DIR__,"..", "..", "get_initial_conditions.jl"))
include(joinpath(@__DIR__,"..", "..", "evaluate_metrics.jl"))


# Get data 
c_scale_data, q_scale_data, input_pretrain, output_pretrain, data_trains, data_full, data_train_eq, AcidPro_ProPro_data, H2O_AcidPro_data, Poh_ProPro_data, AcidPro_ProPro_data_train, H2O_AcidPro_data_train, Poh_ProPro_data_train, AcidPro_ProPro_data_test, H2O_AcidPro_data_test, Poh_ProPro_data_test, AcidPro_ProPro_data_eq, H2O_AcidPro_data_eq, Poh_ProPro_data_eq, c_exp_data3a, c_exp_data3b, c_exp_data4a, c_exp_data4b = get_data()


# Get models and initial conditions 
rhs_tests, x0 = get_initial_conditions()


# Get jacobian prototype to solve ODEs faster 
inlets, outlets, columns, switches, solverOptions = model(dataset = "3", sol_times = data_trains[1][:,end])
p = (columns, columns[1].RHS_q, columns[1].cpp, columns[1].qq, 1, solverOptions.nColumns, solverOptions.idx_units, switches)
jac_proto = sparse(jac_finite_diff(problem!,p,solverOptions.x0 .+ 1e-6, 1e-8))

neurons_test = [2]

# Set selection for training data 
selection = [1,2,3] 

# Initial guesses for Langmuir par
ka = [11.66, 9.04, 2.35, 5.08]
qmax = [9.13, 10.06, 43.07, 5.11]
kl = [0.51, 0.51, 0.51, 0.51]
	
# Run model on training, save training results and plots, run on test data 
NN_model = ComponentArray(ka = ka, qmax = qmax, kl = kl)
u0 = NN_model
evaluate_metrics_training(@__DIR__, 2, 1, 0)

evaluate_metrics_training_all(@__DIR__, 1, 0)


