

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
neurons_test = [2]

# Set selection of k-fold training from 1-5.
selec = [1,2,3,4,5] 

# iter 
global iter = 1
	
# Set up initial conditions for all k fold splits
x0 = get_initial_conditions(selec, columns[1].ConvDispOpInstance.nPoints, columns[1].nComp, qc_exp_data)

# Initial guesses for Langmuir par
ka = [11.66, 5.08]
qmax = [9.13, 5.11]
kl = [0.51, 0.51]


selection = selec
st = 0.0
global u0 = ComponentArray(ka = ka, qmax = qmax, kl = kl)
global p_init_pre = u0

NN_model = ComponentArray(ka = ka, qmax = qmax, kl = kl)
evaluate_metrics_validation(@__DIR__, 1, 2)

# Summmarize results for all tests and save as csv file 
evaluate_metrics_validation_all(@__DIR__)


