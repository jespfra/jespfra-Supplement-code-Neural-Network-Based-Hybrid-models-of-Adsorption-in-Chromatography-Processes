# Hybrid model setup functions required in training and test of model. 
# This jl file should be loaded to run the hybrid model. 


mutable struct hybrid_model{T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11}
	"""
		A struct containing the parameters which are necessary for the hybrid model. 
		The parameters are unpacked whereas the NN parameters are in the p vector to the ODE function. 
		The input to the NN in this setup are scaled c, q 
	"""
    columns::T1
    RHS_q::T2
    cpp::T3
    qq::T4 
    i::T5
    nColumns::T6 
    idx_units::T7
    switches::T8
	solverOptions::T9
	c_scale::T10
	q_scale::T11
end

# Define hybrid RHS 
function (problem!::hybrid_model)(RHS, x, p, t)
	"""
		ODE problem formulation for hybrid models.
	
	"""
    
    @unpack columns, RHS_q, cpp, qq, i, nColumns, idx_units, switches, solverOptions, c_scale, q_scale = problem!
	# i corresponds to section 
	kl = p.kl
	p_NN1 = p.p1
	
	@inbounds for h = 1:nColumns 
		# Compute binding term. 
		# The cpp, qq and rhs_q are set up to ease code reading
		cp1 = (@view x[1 + columns[h].adsStride + idx_units[h] : columns[h].adsStride + columns[h].bindStride * 1 + idx_units[h]]) ./ c_scale
		cp2 = (@view x[1 + columns[h].adsStride + columns[h].bindStride * 1 + idx_units[h] : columns[h].adsStride + columns[h].bindStride * 2 + idx_units[h]]) ./ c_scale
		cp3 = (@view x[1 + columns[h].adsStride + columns[h].bindStride * 2 + idx_units[h] : columns[h].adsStride + columns[h].bindStride * 3 + idx_units[h]]) ./ c_scale
		cp4 = (@view x[1 + columns[h].adsStride + columns[h].bindStride * 3 + idx_units[h] : columns[h].adsStride + columns[h].bindStride * 4 + idx_units[h]]) ./ c_scale
		RHS_q = @view RHS[1 + columns[h].adsStride + columns[h].bindStride * columns[h].nComp + idx_units[h] : columns[h].adsStride + columns[h].bindStride * columns[h].nComp * 2 + idx_units[h]]
		
		#Solid phase residual
		global input = [cp1 cp2 cp3 cp4]'

		denom = 1 .+ (@view nn(input, p_NN1, st)[1][5, 1:end]) .* 7.5 .* cp1 .* c_scale .+ (@view nn(input, p_NN1, st)[1][6, 1:end]) .* 7.5 .* cp2 .* c_scale .+ (@view nn(input, p_NN1, st)[1][7, 1:end]) .* 7.5 .* cp3 .* c_scale .+ (@view nn(input, p_NN1, st)[1][8, 1:end]) .* 7.5 .* cp4 .* c_scale
		
		for j = 1:columns[h].nComp
			RHS_q[1 + (j-1) * columns[h].bindStride : columns[h].bindStride + (j-1) * columns[h].bindStride] .= kl .* (nn(input, p_NN1, st)[1][j, 1:end] .* 50. .* x[1 + columns[h].adsStride + columns[h].bindStride * (j-1) + idx_units[h] : columns[h].adsStride + columns[h].bindStride * j + idx_units[h]] ./ denom .- x[1 + columns[h].adsStride + columns[h].bindStride * columns[h].nComp + idx_units[h] + (j-1) * columns[h].bindStride : columns[h].adsStride + columns[h].bindStride * columns[h].nComp + columns[h].bindStride * j + idx_units[h]])
		end

		# Compute transport term
		compute_transport(RHS, RHS_q, x, columns[h], t, i, h, switches, idx_units)


        # Compute reaction term directly from stationary phase 
		qq = x[1 + columns[h].adsStride + columns[h].bindStride * columns[h].nComp + idx_units[h] : columns[h].adsStride + columns[h].bindStride * columns[h].nComp * 2 + idx_units[h]]
		compute_reaction(RHS_q, qq, columns[h].reaction_solid, columns[h].eps_c, columns[h].nComp, columns[h].bindStride, t)

	end
	nothing
end	


mutable struct hybrid_model_reac{T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11}
	"""
		A struct containing the parameters which are necessary for the hybrid model. 
		The parameters are unpacked whereas the NN parameters are in the p vector to the ODE function. 
		The input to the NN in this setup are scaled c, q 
	"""
    columns::T1
    RHS_q::T2
    cpp::T3
    qq::T4 
    i::T5
    nColumns::T6 
    idx_units::T7
    switches::T8
	solverOptions::T9
	c_scale::T10
	q_scale::T11
end

# Define hybrid RHS 
function (problem!::hybrid_model_reac)(RHS, x, param, t)
	"""
		ODE problem formulation for hybrid models.
	
	"""
    
    @unpack columns, RHS_q, cpp, qq, i, nColumns, idx_units, switches, solverOptions, c_scale, q_scale = problem!
	# i corresponds to section 
	
	kl = param.kl
	p_NN1 = param.p1
	
	@inbounds for h = 1:nColumns 
		# Compute binding term. 
		# The cpp, qq and rhs_q are set up to ease code reading
		cp1 = (@view x[1 + columns[h].adsStride + idx_units[h] : columns[h].adsStride + columns[h].bindStride * 1 + idx_units[h]]) ./ c_scale
		cp2 = (@view x[1 + columns[h].adsStride + columns[h].bindStride * 1 + idx_units[h] : columns[h].adsStride + columns[h].bindStride * 2 + idx_units[h]]) ./ c_scale
		cp3 = (@view x[1 + columns[h].adsStride + columns[h].bindStride * 2 + idx_units[h] : columns[h].adsStride + columns[h].bindStride * 3 + idx_units[h]]) ./ c_scale
		cp4 = (@view x[1 + columns[h].adsStride + columns[h].bindStride * 3 + idx_units[h] : columns[h].adsStride + columns[h].bindStride * 4 + idx_units[h]]) ./ c_scale
		RHS_q = @view RHS[1 + columns[h].adsStride + columns[h].bindStride * columns[h].nComp + idx_units[h] : columns[h].adsStride + columns[h].bindStride * columns[h].nComp * 2 + idx_units[h]]
		
		#Solid phase residual
		global input = [cp1 cp2 cp3 cp4]'

		denom = 1 .+ (@view nn(input, p_NN1, st)[1][5, 1:end]) .* 7.5 .* cp1 .* c_scale .+ (@view nn(input, p_NN1, st)[1][6, 1:end]) .* 7.5 .* cp2 .* c_scale .+ (@view nn(input, p_NN1, st)[1][7, 1:end]) .* 7.5 .* cp3 .* c_scale .+ (@view nn(input, p_NN1, st)[1][8, 1:end]) .* 7.5 .* cp4 .* c_scale
		
		for j = 1:columns[h].nComp
			RHS_q[1 + (j-1) * columns[h].bindStride : columns[h].bindStride + (j-1) * columns[h].bindStride] .= kl .* (nn(input, p_NN1, st)[1][j, 1:end] .* 50. .* x[1 + columns[h].adsStride + columns[h].bindStride * (j-1) + idx_units[h] : columns[h].adsStride + columns[h].bindStride * j + idx_units[h]] ./ denom .- x[1 + columns[h].adsStride + columns[h].bindStride * columns[h].nComp + idx_units[h] + (j-1) * columns[h].bindStride : columns[h].adsStride + columns[h].bindStride * columns[h].nComp + columns[h].bindStride * j + idx_units[h]])
		end

		# Compute transport term
		compute_transport(RHS, RHS_q, x, columns[h], t, i, h, switches, idx_units)


        # Compute reaction term directly from stationary phase 
		qq = x[1 + columns[h].adsStride + columns[h].bindStride * columns[h].nComp + idx_units[h] : columns[h].adsStride + columns[h].bindStride * columns[h].nComp * 2 + idx_units[h]]
		compute_reaction(RHS_q, qq, columns[h].reaction_solid, columns[h].eps_c, columns[h].nComp, columns[h].bindStride, t, input, p_NN1)

	end
	nothing
end	

function pretrain_function(input_pretrain, param, st, h) #param = u0
	denom = 1.0 .+ nn(input_pretrain, param, st)[1][5,:] .* 7.5 .* input_pretrain[1,:] .* c_scale_data[2] .+ nn(input_pretrain, param, st)[1][6,:] .* 7.5 .* input_pretrain[2,:] .* c_scale_data[2] .+ nn(input_pretrain, param, st)[1][7,:] .* 7.5 .* input_pretrain[3,:] .* c_scale_data[2] .+ nn(input_pretrain, param, st)[1][8,:] .* 7.5 .* input_pretrain[4,:] .* c_scale_data[2]
	return nn(input_pretrain, param, st)[1][h,:] .* 50 .* input_pretrain[h,:] .* c_scale_data[2] ./ denom
end
function init_conditions_function(input, params, st, h)
	denom = 1.0 .+ nn(input, params.p1, st)[1][5,:] .* 7.5 .* input[1,:] .* c_scale_data[2] .+ nn(input, params.p1, st)[1][6,:] .* 7.5 .* input[2,:] .* c_scale_data[2] .+ nn(input, params.p1, st)[1][7,:] .* 7.5 .* input[3,:] .* c_scale_data[2] .+ nn(input, params.p1, st)[1][8,:] .* 7.5 .* input[4,:] .* c_scale_data[2]
	return nn(input, params.p1, st)[1][h,:] .* 50 .* input[h,:] .* c_scale_data[2] ./ denom
end

