

# Loss functions 
using DiffEqFlux, Lux, Zygote, ReverseDiff, ComponentArrays
using UnPack


# Loss function 
function loss_general(param)
    # --------------------------Sensealg---------------------------------------------
    sensealg = QuadratureAdjoint(autojacvec = ReverseDiffVJP(true))
    #Interpolating adjoint works well too

    #----------------------------Problem solution-------------------------------------
    loss_total = 0.0

    # sol of first section
	for p = 1:3
		x00 = x0[:,p]
		for h =1:rhs_tests[p].switches.nSections-1

			rhs_tests[p].i = h
			tspan = (rhs_tests[p].switches.section_times[h], rhs_tests[p].switches.section_times[h+1]) 
			fun = ODEFunction(rhs_tests[p]; jac_prototype = jac_proto)
			prob = ODEProblem(fun, x00, (0, tspan[2]-tspan[1]), param)
			idx_1 = findfirst(==(rhs_tests[p].switches.section_times[h]), rhs_tests[p].solverOptions.solution_times)
			idx_2 = findfirst(==(rhs_tests[p].switches.section_times[h+1]), rhs_tests[p].solverOptions.solution_times)
			sol_times = rhs_tests[p].solverOptions.solution_times[idx_1 : idx_2] .- rhs_tests[p].switches.section_times[h]
			sol1 = solve(prob, alg=FBDF(autodiff=false), saveat=sol_times, abstol=rhs_tests[p].solverOptions.abstol, reltol=rhs_tests[p].solverOptions.reltol, sensealg = sensealg) 

			# Determine loss 
			for j = 1:columns[1].nComp
				diff =  abs.(data_trains[p][idx_1:idx_2, j] .-  Array(sol1)[Int(rhs_tests[p].columns[1].ConvDispOpInstance.nPoints*j), 1:end] ./c_scale_data[p]) 
				diffsum = sum(skipmissing(diff))
				loss_total += 1/length(data_trains[p][:, j])^2 * diffsum^2
				
				# stationary phase 
				loss_total += 1/(columns[1].ConvDispOpInstance.nPoints * columns[1].nComp * length(data_train_eq[p][:,j]))^2 * sum((data_train_eq[p][h, j] .- Array(sol1)[Int(1 + columns[1].ConvDispOpInstance.nPoints*(columns[1].nComp + (j-1))) : Int(columns[1].ConvDispOpInstance.nPoints*(columns[1].nComp + j)), end] ./q_scale_data[p]).^2) 
			end
			x00 = sol1.u[end]
		end
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
    l < 1.0e-5
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
    l < 1.0e-5
end



using Optimization, OptimizationOptimisers, OptimizationOptimJL
function opti_nn(;layers = 1)
		
	#------- Maximum Likelihood estimation 
	
	# Set up optimization problem 
	adtype = Optimization.AutoZygote()
	optf = Optimization.OptimizationFunction((x, p) -> loss_general(x), adtype)
	optprob = Optimization.OptimizationProblem(optf, u0)

	if layers == 1
		@time results = Optimization.solve(optprob, OptimizationOptimisers.Adam(0.01), callback = callback1, maxiters = 600)
	elseif layers == 2
		@time results = Optimization.solve(optprob, OptimizationOptimisers.Adam(0.01), callback = callback2, maxiters = 600)
	end
	NN_model = results.u


	
	return NN_model 
end


# pretraining of certain models where the mass transfer coefficients should be around a certain value, y_output
function loss_pretraining(param)

	# Define the mean squared error loss function
	loss_total = 0.0
	for h =1:columns[1].nComp
		loss_total += sum((pretrain_function(input_pretrain, param, st, h) .- output_pretrain[:,h]).^2)
	end
	#println(loss_total)

    return loss_total
end


function opti_nn_pretraining()
	# Set up optimization problem 
	adtype = Optimization.AutoZygote()
	optf = Optimization.OptimizationFunction((x, p) -> loss_pretraining(x), adtype)
	optprob = Optimization.OptimizationProblem(optf, u0_pretrain)
	results = Optimization.solve(optprob, OptimizationOptimisers.Adam(0.01), maxiters = 100)

	return results.u
end
