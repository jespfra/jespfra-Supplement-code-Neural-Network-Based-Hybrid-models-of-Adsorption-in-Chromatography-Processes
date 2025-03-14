
# This file contains the user specified reaction for the hybrid model structure 4.


########################### User specified reaction for hybrid model structure 4 ###########################
mutable struct reaction_HM12 <: reactionBase
	# User implemented reaction 4
	# A special reaction for hybrid model structure 4. 
	# the user input is special for this one when called from the hybrid model structure 4.
	# hence the input is different from the other reactions.
	
	
	# Check parameters
	k1::Float64
	Keq::Float64
	K::Vector{Float64}
	stoich::Vector{Float64}
	gammas::Vector{Float64}
	density::Float64
	Vm::Vector{Float64}
	y_mole::Matrix{Float64} # mole fractions of each component 
	y_mol_tot::Vector{Float64}
	Tmodel # Termodynamic model 
	rate::Vector{Float64}
	idx::UnitRange{Int64}
	T::Float64
	nn
	input
	nn_par 
	st

	
	
	# Define a constructor for Linear that accepts keyword arguments compStride = 80, nComp=4
	function reaction_HM12(; k1::Float64, Keq::Float64, K::Vector{Float64}, stoich::Vector{Float64}, density::Float64, Vm::Vector{Float64}, Tmodel, nComp, compStride, T, nn, nn_par, st)
		
		gammas = ones(Float64, nComp)
		y_mole = zeros(Float64, compStride, nComp)
		y_mol_tot = zeros(Float64, compStride)
		rate = zeros(Float64, compStride)
		idx = 1:compStride
		input = hcat([zeros(Real, compStride)*1. for _ in 1:nComp]...)'
		
		
		new(k1, Keq, K, stoich, gammas, density, Vm, y_mole, y_mol_tot, Tmodel, rate, idx, T, nn, input, nn_par, st)
	end
end



# Compute user specified reaction 
function compute_reaction(RHS_rate, cp_reaction, reaction::reaction_HM12, eps_, nComp, compStride, t, input, param)
	"""
		User implemented reaction from https://doi.org/10.3390/separations9020043
		Takes the concentrations, converts them to molar fractions. 
		Then determines the activity coefficients which are used to determine the reaction rate. 
		Finally, this is added to the stationary phase as the reaction takes place in the stationary phase. 
	"""

	y_mol_tot = cp_reaction[1 : compStride] .* reaction.Vm[1]

	# Determine total mole fraction 
	for j=2:nComp 
		reaction.idx = 1  + (j-1) * compStride  : compStride + (j-1) * compStride
		@views y_mol_tot += reaction.Vm[j] .* cp_reaction[reaction.idx]
	end
	
	# Determine mole fractions for each component 
	y_mole = zeros(eltype(cp_reaction), compStride, nComp)
	rate = zeros(eltype(cp_reaction), compStride)
	for j=1:nComp 
		reaction.idx = 1  + (j-1) * compStride  : compStride + (j-1) * compStride
		y_mole[1:compStride, j] = cp_reaction[reaction.idx] .* reaction.Vm[j] ./ y_mol_tot
	end
	
	for j=1:compStride
		
		# Determine the activity coefficients
		gammas = activity_coefficients_nrtl(reaction.Tmodel, 1e5, reaction.T, (y_mole[j,:]))
	
		# Determine the reaction rate 
		# Ka*Kb*k1(a_1 * a_2 - a_3 * a_4 / K_eq) / (1 + Ka * a_1 + Kb * a_2 + Kc * a_3 + Kd * a_4)
		rate[j] = reaction.k1 * (y_mole[j,1] * gammas[1] * y_mole[j,2] * gammas[2] - y_mole[j,3] * gammas[3] * y_mole[j,4] * gammas[4] / reaction.Keq)
		rate[j] *= reaction.nn(input[:,j], param, reaction.st)[1][1, 1] * 7.5 .* reaction.nn(input[:,j], param, reaction.st)[1][2, 1] * 7.5 / (1 +  reaction.nn(input[:,j], param, reaction.st)[1][1, 1] * 7.5 * y_mole[j,1] * gammas[1]  +  reaction.nn(input[:,j], param, reaction.st)[1][2, 1] * 7.5 * y_mole[j,2] * gammas[2] +  reaction.nn(input[:,j], param, reaction.st)[1][3, 1] * 7.5 * y_mole[j,3] * gammas[3]  +  reaction.nn(input[:,j], param, reaction.st)[1][4, 1] * 7.5 * y_mole[j,4] * gammas[4])
	end
	
	# Add to RHS_rate 
	for j=1:nComp 
		reaction.idx = 1  + (j-1) * compStride  : compStride + (j-1) * compStride
		@. @views RHS_rate[reaction.idx] += reaction.density / (1-eps_) * reaction.stoich[j] * rate[1:compStride]
	end
	
	nothing
end

