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

	p_NN1 = p.p1
	p_NN2 = p.p2
	
	@inbounds for h = 1:nColumns 
		# Compute binding term. 
		# The cpp, qq and rhs_q are set up to ease code reading
		cp1 = (@view x[1 + columns[h].adsStride + idx_units[h] : columns[h].adsStride + columns[h].bindStride * 1 + idx_units[h]]) ./ c_scale
		cp2 = (@view x[1 + columns[h].adsStride + columns[h].bindStride * 1 + idx_units[h] : columns[h].adsStride + columns[h].bindStride * 2 + idx_units[h]]) ./ c_scale
		cp3 = (@view x[1 + columns[h].adsStride + columns[h].bindStride * 2 + idx_units[h] : columns[h].adsStride + columns[h].bindStride * 3 + idx_units[h]]) ./ c_scale
		cp4 = (@view x[1 + columns[h].adsStride + columns[h].bindStride * 3 + idx_units[h] : columns[h].adsStride + columns[h].bindStride * 4 + idx_units[h]]) ./ c_scale
		RHS_q = @view RHS[1 + columns[h].adsStride + columns[h].bindStride * columns[h].nComp + idx_units[h] : columns[h].adsStride + columns[h].bindStride * columns[h].nComp * 2 + idx_units[h]]
		
		#Solid phase residual
		input = [cp1 cp2 cp3 cp4]'

		
		for j = 1:columns[h].nComp
			RHS_q[1 + (j-1) * columns[h].bindStride : columns[h].bindStride + (j-1) * columns[h].bindStride] .= nn2(input, p_NN2, st)[1][1, 1:end] .* (nn(input, p_NN1, st)[1][j, 1:end] .- x[1 + columns[h].adsStride + columns[h].bindStride * columns[h].nComp + idx_units[h] + (j-1) * columns[h].bindStride : columns[h].adsStride + columns[h].bindStride * columns[h].nComp + columns[h].bindStride * j + idx_units[h]])
		end

		# Compute transport term
		compute_transport(RHS, RHS_q, x, columns[h], t, i, h, switches, idx_units)


        # Compute reaction term directly from stationary phase 
		qq = x[1 + columns[h].adsStride + columns[h].bindStride * columns[h].nComp + idx_units[h] : columns[h].adsStride + columns[h].bindStride * columns[h].nComp * 2 + idx_units[h]]
		compute_reaction(RHS_q, qq, columns[h].reaction_solid, columns[h].eps_c, columns[h].nComp, columns[h].bindStride, t)

	end
	nothing
end	


function pretrain_function(input_pretrain, params, st, h)
	return nn(input_pretrain, params, st)[1][h,:]
end
function init_conditions_function(input, params, st, h)
	return nn(input, params.p1, st)[1][h,:]
end


