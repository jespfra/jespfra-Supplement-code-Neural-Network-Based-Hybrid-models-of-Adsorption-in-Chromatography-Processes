

# Loss functions 
using DiffEqFlux, Lux, Zygote, ReverseDiff, ComponentArrays
using UnPack


# Loss function 
function loss_general(param) #param = u0
    # --------------------------Sensealg---------------------------------------------
    sensealg = QuadratureAdjoint(autojacvec = ReverseDiffVJP(true))
    #Interpolating adjoint works well too

    #----------------------------Problem solution-------------------------------------
    loss_total = 0

    # sol of first section
	x00 = x0[:,1]
	for i in selection
		# println(i)
	
		rhs_test.i = i
		tspan = (switches.section_times[i], switches.section_times[i+1]) 
		fun = ODEFunction(rhs_test; jac_prototype = jac_proto)
		prob = ODEProblem(fun, x00, (0, tspan[2]-tspan[1]), param)
		idx_1 = findfirst(==(switches.section_times[i]), solverOptions.solution_times)
		idx_2 = findfirst(==(switches.section_times[i+1]), solverOptions.solution_times)
		sol_times = solverOptions.solution_times[idx_1 : idx_2] .- switches.section_times[i]
		sol1 = solve(prob, alg=FBDF(autodiff=false), saveat=sol_times, abstol=solverOptions.abstol, reltol=solverOptions.reltol, sensealg = sensealg) 

		# Determine loss j=1
		for j = 1: columns[1].nComp
			loss_total += 1/length(data_train_c[1])^2 * sum(abs2, (data_train_c[j][idx_1:idx_2] .- Array(sol1)[Int(columns[1].ConvDispOpInstance.nPoints*j), 1:end] ./c_scale_data))
		end
		x00 = sol1.u[end]
	end



    #----------------------------Output---------------------------------------------
    # output total loss
    return loss_total
end


callback1 = function(state, l) 
	# Read the existing CSV file
    convergence = CSV.read(joinpath(saveat, "convergence.csv"), DataFrame)
    
    # Append the new data
	global iter
    push!(convergence, (iter, l))
	iter += 1
    
    # Save the updated DataFrame back to the CSV file
    CSV.write(joinpath(saveat, "convergence.csv"), convergence)
    
    # Increment the iteration counter
    l < 1.0e-6
end

callback2 = function(state, l) 
	# Read the existing CSV file
    convergence = CSV.read(joinpath(saveat, "convergence_2.csv"), DataFrame)
    
    # Append the new data
	global iter
    push!(convergence, (iter, l))
	iter += 1
    
    # Save the updated DataFrame back to the CSV file
    CSV.write(joinpath(saveat, "convergence_2.csv"), convergence)
    
    # Increment the iteration counter
    l < 1.0e-6
end



using Optimization, OptimizationOptimisers, OptimizationOptimJL
function opti_nn(;layers = 1)
		
	#------- Maximum Likelihood estimation 
	
	# Set up optimization problem 
	adtype = Optimization.AutoZygote()
	optf = Optimization.OptimizationFunction((x, p) -> loss_general(x), adtype)
	optprob = Optimization.OptimizationProblem(optf, u0)

	if layers == 1
		@time results = Optimization.solve(optprob, OptimizationOptimisers.Adam(0.01), callback = callback1, maxiters = 300)
	elseif layers == 2
		@time results = Optimization.solve(optprob, OptimizationOptimisers.Adam(0.01), callback = callback2, maxiters = 300)
	end
	NN_model = results.u

	# optf2 = Optimization.OptimizationFunction((x, p) -> loss2(x), adtype)
	# optprob2 = Optimization.OptimizationProblem(optf2, results.u)

	# @time results_2 = Optimization.solve(optprob2, Optim.BFGS(initial_stepnorm = 0.01), 
	# callback = callback, maxiters = 75, maxtime = 35*60, allow_f_increases = true)

	# NN_model = results_2.u
	
	return NN_model 
end

# pretraining of certain models where the mass transfer coefficients should be around a certain value, y_output
function loss_pretraining(params)

	# Define the mean squared error loss function
	loss_total = 0.0
	for h =1:columns[1].nComp
		loss_total += sum((pretrain_function(input_pretrain, params, st, h) .- output_pretrain[:,h]).^2)
	end

    return loss_total
end


function opti_nn_pretraining()
	# Set up optimization problem 
	adtype = Optimization.AutoZygote()
	optf = Optimization.OptimizationFunction((x, p) -> loss_pretraining(x), adtype)
	optprob = Optimization.OptimizationProblem(optf, u0_pretrain)
	results = Optimization.solve(optprob, OptimizationOptimisers.Adam(0.01), maxiters = 200)

	return results.u
end
