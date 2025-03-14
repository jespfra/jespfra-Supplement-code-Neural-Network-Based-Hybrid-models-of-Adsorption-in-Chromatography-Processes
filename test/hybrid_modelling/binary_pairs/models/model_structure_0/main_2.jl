

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


# Neurons tested
# neurons_test = [9] #collect(2:1:10)

# Set selection for training data 
selection = [1,2,3] 
	


for l in neurons_test #l=2
	println("Training at $l neurons")

	# Set up neural network - input and output depends on model structure
	global nn, p_init, st = initialize_neural_network(saveat = joinpath(@__DIR__, "neurons_$(l)_2"), input = 8, neurons = [l,l], output = 4, activation = (tanh_fast, tanh_fast, identity))

	# Inital weights - vary if using multiple simulatenous networks in one model and/or mechanistic parameters
	global u0 =  ComponentArray(p_init)
	u0.layer_1.weight .*= 1e-6
	u0.layer_2.weight .*= 1e-6
	
	
	# Run optimization on weights #NN_model=p_init
	global saveat = joinpath(@__DIR__, "neurons_$(l)_2")
	global iter = 1 # Reset neuron
	NN_model = opti_nn(layers = 2,epochs = 900)
	
	# save model 
	@save joinpath(@__DIR__, "neurons_$(l)_2","NN_model.jld2") NN_model
	global NN_model = load(joinpath(@__DIR__, "neurons_$(l)_2","NN_model.jld2"))["NN_model"]
	
	# Run model on training, save training results and plots, run on test data 
	evaluate_metrics_training(@__DIR__, l, 2, 0)

end
# Neurons tested
neurons_test = collect(2:1:10)

# Summmarize results for all tests and save as csv file 
#evaluate_metrics_training_all(@__DIR__, 2)

# Train most promising model on all data 
#df_avg = CSV.read(joinpath(@__DIR__, "metrics_train_avg_2.csv"), DataFrame)
#idx_min = findfirst(x -> x == minimum(df_avg.MSE_test), df_avg.MSE_test)
#nn, p_init, st = initialize_neural_network(saveat = @__DIR__, input = 4, neurons = df_avg.neurons[idx_min], output = 4, activation = (tanh_fast, tanh_fast, softplus))
#NN_model = load(joinpath(@__DIR__, "neurons_$(df_avg.neurons[idx_min])_2","NN_model.jld2"))["NN_model"]

# save model 
#@save joinpath(@__DIR__, "train_best", "NN_model_best_2.jld2") NN_model

# Run model on test and save test results and plots
#evaluate_metrics_all(@__DIR__, 2)


