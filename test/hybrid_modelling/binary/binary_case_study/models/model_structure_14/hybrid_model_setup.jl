# Hybrid model setup functions required in training and test of model. 
# This jl file should be loaded to run the hybrid model. 


mutable struct hybrid_model{T1, T2, T3, T4, T5, T6, T7, T8, T9, T10}
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
	c_scale::T9
	q_scale::T10
end

# Define hybrid RHS 
function (problem!::hybrid_model)(RHS, x, p, t)
	"""
		ODE problem formulation for hybrid models.
	
	"""
    
    @unpack columns, RHS_q, cpp, qq, i, nColumns, idx_units, switches, c_scale, q_scale = problem!
	# i corresponds to section 
	p_NN1 = p.p1
	kl = p.kl
	qmax = p.qmax

	
	@inbounds for h = 1:nColumns 
		# Compute binding term. 
		# The cpp, qq and rhs_q are set up to ease code reading
		cp1 = (@view x[1 + columns[h].adsStride + idx_units[h] : columns[h].adsStride + columns[h].bindStride * 1 + idx_units[h]]) ./ c_scale
		cp2 = (@view x[1 + columns[h].adsStride + columns[h].bindStride * 1 + idx_units[h] : columns[h].adsStride + columns[h].bindStride * 2 + idx_units[h]]) ./ c_scale
		RHS_q = @view RHS[1 + columns[h].adsStride + columns[h].bindStride * columns[h].nComp + idx_units[h] : columns[h].adsStride + columns[h].bindStride * columns[h].nComp * 2 + idx_units[h]]
		
		#Solid phase residual
        q1 = (@view x[1 + columns[h].adsStride + columns[h].bindStride * columns[h].nComp + idx_units[h] : columns[h].adsStride + columns[h].bindStride * columns[h].nComp + columns[h].bindStride * 1 + idx_units[h]]) 
		q2 = (@view x[1 + columns[h].adsStride + columns[h].bindStride * columns[h].nComp + idx_units[h] + columns[h].bindStride * 1 : columns[h].adsStride + columns[h].bindStride * columns[h].nComp + columns[h].bindStride * 2 + idx_units[h]])
        input = [cp1 cp2]'
		denom = 1 .+ (@view nn(input, p_NN1, st)[1][1, 1:end]) .* 7.5 .* cp1 .* c_scale .+ (@view nn(input, p_NN1, st)[1][2, 1:end]) .* 7.5 .* cp2 .* c_scale
		RHS_q[1 : columns[h].bindStride] .= kl[1] .* (qmax[1] .* (@view nn(input, p_NN1, st)[1][1, 1:end]) .* 7.5 .*cp1 .* c_scale ./ denom .- q1)
		RHS_q[1 + columns[h].bindStride : columns[h].bindStride * 2] .= kl[2] .* (qmax[2] .* (@view nn(input, p_NN1, st)[1][2, 1:end]) .* 7.5 .* cp2 .* c_scale ./ denom .- q2)

		# Compute transport term
		compute_transport(RHS, RHS_q, x, columns[h], t, i, h, switches, idx_units)

        # println(t)

	end
	nothing
end	

function pretrain_function(input_pretrain, params, st, h)
	denom = 1.0 .+ nn(input_pretrain, params, st)[1][1,:] .* 7.5 .* input_pretrain[1,:] .* c_scale_data .+ nn(input_pretrain, params, st)[1][2,:] .* 7.5 .* input_pretrain[2,:] .* c_scale_data
	return qmax[h] .* nn(input_pretrain, params, st)[1][h,:] .* 7.5 .* input_pretrain[h,:] .* c_scale_data ./ denom
end
function init_conditions_function(input, params, st, h)
	denom = 1.0 .+ nn(input, params.p1, st)[1][1,:] .* 7.5 .* input[1,:] .* c_scale_data .+ nn(input, params.p1, st)[1][2,:] .* 7.5 .* input[2,:] .* c_scale_data
	return params.qmax[h] .*nn(input, params.p1, st)[1][h,:] .* 7.5 .* input[h,:] .* c_scale_data ./ denom
end


