



# Set flags if reaction is activated or deactivated 
# If deactivated, ReactionState will take NoReactions and then do nothing when called. 
# If activated, ReactionState will take YesReactions and determine the user specified reaction. 
abstract type ReactionState end

struct NoReactions <: ReactionState end
struct YesReactions <: ReactionState end
struct CSTRReactions <: ReactionState end


function compute_reactions!(RHS, RHS_q, cpp, qq, x, idx_units, m, t, checkReactions::NoReactions)
	"""
	 If there are no reactions, continue 
	"""
	
	nothing
end

function compute_reactions!(RHS, RHS_q, cpp, qq, x, idx_units_jump, m, t, checkReactions::YesReactions)
	"""
	 If there is one or more reactions, compute reactions in the liquid bulk phase, liquid particle phase and solid phase. 
	"""
	
	# Solid phase reactions 
	compute_reaction!(RHS_q, qq, m.reaction_solid, m.eps_c, m.nComp, m.bindStride, t)
	
	# Particle phase reactions 
	RHS_q = @view RHS[1 + m.adsStride + idx_units_jump : m.adsStride + m.bindStride * m.nComp + idx_units_jump]
	compute_reaction!(RHS_q, cpp, m.reaction_particle, m.eps_c, m.nComp, m.bindStride, t)
	
	# Liquid phase reactions 
	cpp = @view x[1 + idx_units_jump : m.ConvDispOpInstance.nPoints * m.nComp + idx_units_jump]
	RHS_q = @view RHS[1 + idx_units_jump : m.ConvDispOpInstance.nPoints * m.nComp + idx_units_jump]
	compute_reaction!(RHS_q, cpp, m.reaction_liquid, m.eps_c, m.nComp, m.ConvDispOpInstance.nPoints * m.nComp, t)
	
	nothing
end

function compute_reactions!(RHS, RHS_q, cpp, qq, x, idx_units_jump, m, t, checkReactions::CSTRReactions)
	"""
	 If the type is CSTR and there is a reaction, compute the CSTR reaction
	"""
	
	# CSTR Reactions 
	RHS_q = @view RHS[1 + idx_units_jump : m.nComp + idx_units_jump]
	cpp =  @view x[1 + idx_units_jump : m.nComp + idx_units_jump]
	compute_reaction!(RHS_q, cpp, m.reaction_cstr, 1, m.nComp, 1, t)
	
	nothing
end




# Define reactionBase
abstract type reactionBase 
	# Here the binding is specified

end

########################### no reaction ###########################
mutable struct no_reaction <: reactionBase
	# Is the default - no reaction taking place
	# Check parameters
	# Define a constructor for Linear that accepts keyword arguments
	# function no_reaction()
	# 	new(ka)
	# end
end

# Computing the default no reaction
function compute_reaction!(RHS_rate, cp_reaction, reaction::no_reaction, eps_, nComp, compStride, t)
	"""
		Default no reaction taking place, does nothing. 
	"""
	nothing
end
# Computing the default no reaction
function compute_reaction(RHS_rate, cp_reaction, reaction::no_reaction, eps_, nComp, compStride, t)
	"""
		Default no reaction taking place, does nothing. 
	"""
	nothing
end


########################### User specified reaction 1 ###########################
mutable struct reaction_1 <: reactionBase
	# User implemented reaction 1
	
	
	# Check parameters
	k1::Float64
	Keq::Float64
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
	
	
	# Define a constructor for Linear that accepts keyword arguments
	function reaction_1(; k1::Float64, Keq::Float64, stoich::Vector{Float64}, density::Float64, Vm::Vector{Float64}, Tmodel, nComp, compStride, T)
		
		gammas = ones(Float64, nComp)
		y_mole = zeros(Float64, compStride, nComp)
		y_mol_tot = zeros(Float64, compStride)
		rate = zeros(Float64, compStride)
		idx = 1:compStride
		
		
		new(k1, Keq, stoich, gammas, density, Vm, y_mole, y_mol_tot, Tmodel, rate, idx, T)
	end
end


# Compute user specified reaction 
function compute_reaction!(RHS_rate, cp_reaction, reaction::reaction_1, eps_, nComp, compStride, t)
	"""
		User implemented reaction from https://doi.org/10.1016/j.compchemeng.2019.06.010
		Takes the concentrations, converts them to molar fractions. 
		Then determines the activity coefficients which are used to determine the reaction rate. 
		Finally, this is added to the stationary phase as the reaction takes place in the stationary phase. 
	"""
	fill!(reaction.y_mol_tot, 0.0)

	# Determine total mole fraction 
	for j=1:nComp 
		reaction.idx = 1  + (j-1) * compStride  : compStride + (j-1) * compStride
		@. @views reaction.y_mol_tot += reaction.Vm[j] * cp_reaction[reaction.idx]
	end
	
	# Determine mole fractions for each component 
	for j=1:nComp 
		reaction.idx = 1  + (j-1) * compStride  : compStride + (j-1) * compStride
		@views reaction.y_mole[1:compStride, j] = cp_reaction[reaction.idx] .* reaction.Vm[j] ./ reaction.y_mol_tot
	end
	
	for j=1:compStride
		
		# Determine the activity coefficients
		reaction.gammas .= activity_coefficients_nrtl(reaction.Tmodel, 1e5, reaction.T, @view(reaction.y_mole[j,:]) )
	
	
		# Determine the reaction rate 
		# k1(a_1 * a_2 - a_3 * a_4 / K_eq)
		reaction.rate[j] = reaction.k1 * (reaction.y_mole[j,1] * reaction.gammas[1] * reaction.y_mole[j,2] * reaction.gammas[2] - reaction.y_mole[j,3] * reaction.gammas[3] * reaction.y_mole[j,4] * reaction.gammas[4] / reaction.Keq)
	end
	
	# Add to RHS_rate 
	for j=1:nComp 
		reaction.idx = 1  + (j-1) * compStride  : compStride + (j-1) * compStride
		@. @views RHS_rate[reaction.idx] += reaction.density / (1-eps_) * reaction.stoich[j] * reaction.rate[1:compStride]
	end
	
	nothing
end


function compute_reaction(RHS_rate, cp_reaction, reaction::reaction_1, eps_, nComp, compStride, t)
	"""
		User implemented reaction from https://doi.org/10.1016/j.compchemeng.2019.06.010
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
		# k1(a_1 * a_2 - a_3 * a_4 / K_eq)
		rate[j] = reaction.k1 * (y_mole[j,1] * gammas[1] *y_mole[j,2] * gammas[2] - y_mole[j,3] * gammas[3] * y_mole[j,4] * gammas[4] / reaction.Keq)
	end
	
	# Add to RHS_rate 
	for j=1:nComp 
		reaction.idx = 1  + (j-1) * compStride  : compStride + (j-1) * compStride
		@. @views RHS_rate[reaction.idx] += reaction.density / (1-eps_) * reaction.stoich[j] * rate[1:compStride]
	end
	
	nothing
end




########################### User specified reaction 2 ###########################
mutable struct reaction_1_cstr <: reactionBase
	# User implemented reaction 1
	
	
	# Check parameters
	k1::Float64
	Keq::Float64
	stoich::Vector{Float64}
	gammas::Vector{Float64}
	Vm::Vector{Float64}
	y_mole::Vector{Float64} # mole fractions of each component 
	y_mol_tot::Vector{Float64}
	Tmodel # Termodynamic model 
	rate::Vector{Float64}
	idx::UnitRange{Int64}
	active_sites::Float64 #area 
	m_cat_dry::Float64 # dry catalyst matter 
	N_t::Float64 # total moles
	T::Float64
	
	
	# Define a constructor for Linear that accepts keyword arguments
	function reaction_1_cstr(; k1::Float64, Keq::Float64, stoich::Vector{Float64}, Vm::Vector{Float64}, Tmodel, nComp, compStride, active_sites, m_cat_dry, N_t, T)
		
		gammas = zeros(Float64, nComp)
		y_mole = zeros(Float64, nComp)
		y_mol_tot = zeros(Float64, compStride)
		rate = zeros(Float64, nComp)
		idx = 1:compStride
		
		
		new(k1, Keq, stoich, gammas, Vm, y_mole, y_mol_tot, Tmodel, rate, idx, active_sites, m_cat_dry, N_t, T)
	end
end


# Compute user specified reaction 

function compute_reaction!(RHS_rate, cp_reaction, reaction::reaction_1_cstr, eps_, nComp, compStride, t)
	"""
		User implemented reaction from https://doi.org/10.1016/j.compchemeng.2019.06.010
		Takes the concentrations, converts them to molar fractions. 
		Then determines the activity coefficients which are used to determine the reaction rate. 
		Finally, this is added to the stationary phase as the reaction takes place in the stationary phase.
		Input for this reaction is in mole fraction.  
	"""
	
	
	# Determine mole fractions for each component 
	for j=1:nComp 
		reaction.y_mole[j] = cp_reaction[j] #./ reaction.y_mol_tot[1]
	end
	reaction.gammas = activity_coefficients_nrtl(reaction.Tmodel, 1e5, reaction.T, reaction.y_mole)
	# k1(a_1 * a_2 - a_3 * a_4 / K_eq)
	reaction.rate[1] = reaction.k1 * (reaction.y_mole[1] * reaction.gammas[1] * reaction.y_mole[2] * reaction.gammas[2] - reaction.y_mole[3] * reaction.gammas[3] * reaction.y_mole[4] * reaction.gammas[4] / reaction.Keq)
	
	# Add to RHS_rate 
	for j=1:nComp 
		RHS_rate[j] += reaction.active_sites * reaction.m_cat_dry * reaction.stoich[j] * reaction.rate[1] / reaction.N_t 
	end
	# println(RHS_rate[1])
	
	nothing
end



########################### User specified reaction 2 ###########################
mutable struct reaction_2 <: reactionBase
	# User implemented reaction 1
	
	
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
	
	
	# Define a constructor for Linear that accepts keyword arguments
	function reaction_2(; k1::Float64, Keq::Float64, K::Vector{Float64}, stoich::Vector{Float64}, density::Float64, Vm::Vector{Float64}, Tmodel, nComp, compStride, T)
		
		gammas = ones(Float64, nComp)
		y_mole = zeros(Float64, compStride, nComp)
		y_mol_tot = zeros(Float64, compStride)
		rate = zeros(Float64, compStride)
		idx = 1:compStride
		
		
		new(k1, Keq, K, stoich, gammas, density, Vm, y_mole, y_mol_tot, Tmodel, rate, idx, T)
	end
end


# Compute user specified reaction 
function compute_reaction!(RHS_rate, cp_reaction, reaction::reaction_2, eps_, nComp, compStride, t)
	"""
		User implemented reaction from https://doi.org/10.3390/separations9020043
		Takes the concentrations, converts them to molar fractions. 
		Then determines the activity coefficients which are used to determine the reaction rate. 
		Finally, this is added to the stationary phase as the reaction takes place in the stationary phase. 
	"""
	fill!(reaction.y_mol_tot, 0.0)

	# Determine total mole fraction 
	for j=1:nComp 
		reaction.idx = 1  + (j-1) * compStride  : compStride + (j-1) * compStride
		@. @views reaction.y_mol_tot += reaction.Vm[j] * cp_reaction[reaction.idx]
	end
	
	# Determine mole fractions for each component 
	for j=1:nComp 
		reaction.idx = 1  + (j-1) * compStride  : compStride + (j-1) * compStride
		@views reaction.y_mole[1:compStride, j] = cp_reaction[reaction.idx] .* reaction.Vm[j] ./ reaction.y_mol_tot
	end
	
	for j=1:compStride
		
		# Determine the activity coefficients
		reaction.gammas .= activity_coefficients_nrtl(reaction.Tmodel, 1e5, reaction.T, @view(reaction.y_mole[j,:]) )
	
	
		# Determine the reaction rate 
		# Ka*Kb*k1(a_1 * a_2 - a_3 * a_4 / K_eq) / (1 + Ka * a_1 + Kb * a_2 + Kc * a_3 + Kd * a_4)
		reaction.rate[j] = (reaction.k1 * (reaction.y_mole[j,1] * reaction.gammas[1] * reaction.y_mole[j,2] * reaction.gammas[2] - reaction.y_mole[j,3] * reaction.gammas[3] * reaction.y_mole[j,4] * reaction.gammas[4] / reaction.Keq))
		reaction.rate[j] *= reaction.K[1]*reaction.K[2] / (1 + reaction.K[1] * reaction.y_mole[j,1] * reaction.gammas[1]  + reaction.K[2] * reaction.y_mole[j,2] * reaction.gammas[2] + reaction.K[3] * reaction.y_mole[j,3] * reaction.gammas[3]  + reaction.K[4] * reaction.y_mole[j,4] * reaction.gammas[4])
	end
	
	# Add to RHS_rate 
	for j=1:nComp 
		reaction.idx = 1  + (j-1) * compStride  : compStride + (j-1) * compStride
		@. @views RHS_rate[reaction.idx] += reaction.density / (1-eps_) * reaction.stoich[j] * reaction.rate[1:compStride]
	end
	
	nothing
end


function compute_reaction(RHS_rate, cp_reaction, reaction::reaction_2, eps_, nComp, compStride, t)
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
		gammas = activity_coefficients_nrtl(reaction.Tmodel, 1e5, reaction.T, y_mole[j,:])
	
		# Determine the reaction rate 
		# k1(a_1 * a_2 - a_3 * a_4 / K_eq)
		rate[j] = reaction.k1 * (y_mole[j,1] * gammas[1] * y_mole[j,2] * gammas[2] - y_mole[j,3] * gammas[3] * y_mole[j,4] * gammas[4] / reaction.Keq)
		rate[j] *= reaction.K[1]*reaction.K[2] / (1 + reaction.K[1] * y_mole[j,1] * gammas[1]  + reaction.K[2] * y_mole[j,2] * gammas[2] + reaction.K[3] * y_mole[j,3] * gammas[3]  + reaction.K[4] * y_mole[j,4] * gammas[4])
	end
	
	# Add to RHS_rate 
	for j=1:nComp 
		reaction.idx = 1  + (j-1) * compStride  : compStride + (j-1) * compStride
		@. @views RHS_rate[reaction.idx] += reaction.density / (1-eps_) * reaction.stoich[j] * rate[1:compStride]
	end
	
	nothing
end


