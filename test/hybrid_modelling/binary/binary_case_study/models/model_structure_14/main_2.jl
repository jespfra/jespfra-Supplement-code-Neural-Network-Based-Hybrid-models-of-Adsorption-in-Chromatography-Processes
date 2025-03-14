

# Add the include file custom to load packages and scripts. 
# the file is located on the main from which the file takes care of the rest. 
include(joinpath(@__DIR__,"..","..","..","..","..","..","include.jl"))
include(joinpath(@__DIR__,"..","..","..","..","include_hybrid.jl"))


############## Solve using a hybrid modelling approach ##############
include(joinpath(@__DIR__,"..", "..", "loss_functions.jl"))
include(joinpath(@__DIR__, "hybrid_model_setup.jl"))
include(joinpath(@__DIR__,"..", "..", "lrm_model.jl"))
# include(joinpath(@__DIR__,".." ,"plot_functions.jl"))
include(joinpath(@__DIR__,"..", "..", "get_data.jl"))
include(joinpath(@__DIR__,"..", "..", "get_initial_conditions.jl"))
include(joinpath(@__DIR__,"..", "..", "evaluate_metrics.jl"))


# Get data 
test_split = 204
c_exp_data, qc_exp_data, input_pretrain, output_pretrain, data_train_c, data_train_q, data_test_c, data_test_q, c_scale_data, q_scale_data = get_data(test_split=test_split)

	
# Get jacobian prototype to solve ODEs faster 
inlets, outlets, columns, switches, solverOptions = model()
rhs_test = hybrid_model(columns, columns[1].RHS_q, columns[1].cpp, columns[1].qq, 1, solverOptions.nColumns, solverOptions.idx_units, switches, c_scale_data, q_scale_data)
p = (columns, columns[1].RHS_q, columns[1].cpp, columns[1].qq, 1, solverOptions.nColumns, solverOptions.idx_units, switches)
jac_proto = sparse(jac_finite_diff(problem!,p,solverOptions.x0 .+ 1e-6, 1e-8))



# Neurons tested
neurons_test = collect(2:1:10)

# Set selection of k-fold training from 1-5.
selec = [1,2,3,4,5] 

# iter 
global iter = 1
	
# Set up initial conditions for all k fold splits
x0 = get_initial_conditions(selec, columns[1].ConvDispOpInstance.nPoints, columns[1].nComp, qc_exp_data)

# Linear driving force constant
kl = [0.25, 0.25]
qmax = [12.0, 6.0]

for l in neurons_test #l=2
	println("Training at $l neurons")

	# Set up neural network - input and output depends on model structure
	global nn, p_init, st = initialize_neural_network(saveat = joinpath(@__DIR__, "neurons_$(l)_2"), input = 2, neurons = [l,l], output = 2, activation = (tanh_fast, tanh_fast, softplus))

	# Inital weights - vary if using multiple simulatenous networks in one model and/or mechanistic parameters
	global u0_pretrain = ComponentArray(p_init)

	# pretraing model
	global p_init_pre = opti_nn_pretraining()

	# Inital weights - vary if using multiple simulatenous networks in one model and/or mechanistic parameters
	global u0 = ComponentArray(kl = kl, qmax = qmax, p1 = p_init_pre)
	
	
	# optimization on weights
	selection = copy(selec)
	global selection
	
	# Run optimization on weights 
	global saveat = joinpath(@__DIR__, "neurons_$(l)_2")
	global iter = 1 # Reset iter for each k-fold
	NN_model = opti_nn(layers = 2)
	
	# save model 
	@save joinpath(@__DIR__, "neurons_$(l)_2","NN_model.jld2") NN_model
	global NN_model = load(joinpath(@__DIR__, "neurons_$(l)_2","NN_model.jld2"))["NN_model"]
	
	# Run model on validation and save validation results and plots
	evaluate_metrics_validation(@__DIR__, 1, l, 2)

end

neurons_test = collect(2:1:10)
# Summmarize results for all tests and save as csv file 
evaluate_metrics_validation_all(@__DIR__, 2)

# Train most promising model on all data 
# Run optimization on weights 
df_avg = CSV.read(joinpath(@__DIR__, "metrics_train_avg_2.csv"), DataFrame)
idx_min = findfirst(x -> x == minimum(df_avg.MSE_test), df_avg.MSE_test)
NN_model = load(joinpath(@__DIR__, "neurons_$(df_avg.neurons[idx_min])_2","NN_model.jld2"))["NN_model"]

# save model 
@save joinpath(@__DIR__, "train_best", "NN_model_best_2.jld2") NN_model


# Run model on validation and save validation results and plots
evaluate_metrics_all(@__DIR__, 2)


