
function activity_coefficients_nrtl(model, P, T, z)
    """

    NRTL multicomponent activity coefficient model. 
    Parameters are stored in model. 
    P is the pressure, in this case irrelevant. 
    T is the Temperature in K. 
    z are the molar compositions. 
    
    """

    # Extract parameters from the model
    a, b, c = model
    
    # Normalize mole fractions
    x = z ./ sum(z)
    
    # Calculate tau and G matrices
    tau = @. a + b / T
    G = @. exp(-c * tau)
    
    # Number of components
    nComp = size(a)[1] # rewrite using mutable struct 
    
    # Initialize lnγ array
    lny = zeros(eltype(z), nComp) # 
    
    # Calculate lnγ for each component
    for i in 1:nComp
        sum1 = 0.0
        sum2 = 0.0
        for j in 1:nComp
            sum1 += x[j] * tau[j, i] * G[j, i]
            sum2 += x[j] * G[j, i]
        end
        term1 = sum1 / sum2 # sum(x[j]*τ[j,:].*G[j,:] for j ∈ @comps)./sum(x[k]*G[k,:] for k ∈ @comps)
        
        sum3 = 0.0
        for j in 1:nComp
            sum4 = 0.0
            sum5 = 0.0
            for k in 1:nComp
                sum4 += x[k] * G[k, j]
                sum5 += x[k] * tau[k, j] * G[k, j]
            end
            term2 = G[i, j] / sum4 * (tau[i, j] - sum5 / sum4)
            sum3 += x[j] * term2 #sum(x[j]*G[:,j]/sum(x[k]*G[k,j] for k ∈ @comps).*(τ[:,j] .-sum(x[m]*τ[m,j]*G[m,j] for m ∈ @comps)/sum(x[k]*G[k,j] for k ∈ @comps)) for j in @comps)
        end
        
        lny[i] += term1 + sum3
    end
    
    # Return the exponential of lnγ
    return exp.(lny)
end