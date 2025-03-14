

# Get initial conditions  
function get_initial_conditions(selec, nPoints, nComp, qc_exp_data)
	x0 = zeros(Float64, 2*nPoints*nComp, length(selec))

	for i =1:length(selec)
		for j in 1:nComp #j=1
			x0[1 + nPoints*(j-1) : j*nPoints, i] .= qc_exp_data[!,"C$(j-1)"][i]
			x0[1 + nPoints*nComp + nPoints*(j-1) : nPoints*nComp + j*nPoints, i] .= qc_exp_data[!,"q$(j-1)"][i]
		end 
	end
	return x0 
end 
		
