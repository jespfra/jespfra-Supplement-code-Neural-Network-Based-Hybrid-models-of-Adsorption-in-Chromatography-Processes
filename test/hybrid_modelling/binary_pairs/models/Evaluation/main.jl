

# the file is located on the main from which the file takes care of the rest. 
include(joinpath(@__DIR__,"..","..","..","..","..","include.jl"))
include(joinpath(@__DIR__,"..","..","..","include_hybrid.jl"))


############## Solve using a hybrid modelling approach ##############
include(joinpath(@__DIR__,"..", "..", "loss_functions.jl"))
include(joinpath(@__DIR__, "hybrid_model_setup.jl"))
include(joinpath(@__DIR__,"..", "..", "lrm_models.jl"))
include(joinpath(@__DIR__,"..", "..", "get_data.jl"))
include(joinpath(@__DIR__,"..", "..", "get_initial_conditions.jl"))
include(joinpath(@__DIR__,"..", "..", "evaluate_metrics.jl"))
using LaTeXStrings

# Get data 
c_scale_data, q_scale_data, input_pretrain, output_pretrain, data_trains, data_full, data_train_eq, AcidPro_ProPro_data, H2O_AcidPro_data, Poh_ProPro_data, AcidPro_ProPro_data_train, H2O_AcidPro_data_train, Poh_ProPro_data_train, AcidPro_ProPro_data_test, H2O_AcidPro_data_test, Poh_ProPro_data_test, AcidPro_ProPro_data_eq, H2O_AcidPro_data_eq, Poh_ProPro_data_eq, c_exp_data3a, c_exp_data3b, c_exp_data4a, c_exp_data4b = get_data()


## evaluation of the mechanistic model and the best performing hybrid model. 

# first plotting the mechanistic model
# solving the mechanistic model 
selection = [1,2,3] 

# Initial guesses for Langmuir par
ka = [11.66, 9.04, 2.35, 5.08]
qmax = [9.13, 10.06, 43.07, 5.11]
kl = [0.51, 0.51, 0.51, 0.51]
	
# Run model on training, save training results and plots, run on test data 
NN_model_mm = ComponentArray(ka = ka, qmax = qmax, kl = kl)


# impoting the best performing hybrid model
structure_best = 15
layers = 2
structure_best_plot = 10
include(joinpath(@__DIR__, "..", "model_structure_$structure_best", "hybrid_model_setup.jl"))
if structure_best == 4 || structure_best == 12
    include(joinpath(@__DIR__, "..", "model_structure_$structure_best", "reaction_HM$structure_best.jl"))
end
df_avg = CSV.read(joinpath(@__DIR__,  "..", "model_structure_$(structure_best)", "metrics_train_avg_$layers.csv"), DataFrame)
idx_min = findfirst(x -> x == minimum(df_avg.MSE_test), df_avg.MSE_test)
NN_model = load(joinpath(@__DIR__, "..", "model_structure_$structure_best", "neurons_$(df_avg.neurons[idx_min])_$layers","NN_model.jld2"))["NN_model"]
nn, p_init, st = initialize_neural_network(saveat = joinpath(@__DIR__), input = 4, neurons = [df_avg.neurons[idx_min],df_avg.neurons[idx_min]], output = 4, activation = (tanh_fast, tanh_fast, softplus))

# Plotting the binary pairs data 
colors = ["blue", "green", "purple", "red"]
colors_mm = ["magenta", "brown", "orange", "olive"]
components = ["POH", "AcidPro", "H2O", "ProPro"]
for p=1:3 #p=1
    
    # Solving the mechanistic model 
    sol_times = data_full[p][:,end]
    test_split = findfirst(x -> x == data_trains[p][end,end], sol_times) 
    sol_times_mm = sort(union(collect(0:0.1:data_trains[p][end,end]), sol_times))
    test_split_refined = findfirst(x -> x == data_trains[p][end,end], sol_times_mm)
    inlets, outlets, columns_mm, switches, solverOptions = model(dataset = "$p", sol_times = sol_times_mm)
    rhs_test = hybrid_model_mm(columns_mm, columns_mm[1].RHS_q, columns_mm[1].cpp, columns_mm[1].qq, 1, solverOptions.nColumns, solverOptions.idx_units, switches, solverOptions, c_scale_data[2], 1.0)
    solve_model_hybrid(columns=columns_mm, switches=switches, solverOptions=solverOptions, hybrid_model_setup=rhs_test, p_NN=NN_model_mm, outlets=outlets, alg=FBDF(autodiff=false))


    # Solving the hybrid model 
    inlets, outlets, columns, switches, solverOptions = model(dataset = "$p", sol_times=sol_times_mm)
    input_init = [solverOptions.x0[1 + (1-1) * columns[1].ConvDispOpInstance.nPoints : 1 * columns[1].ConvDispOpInstance.nPoints] solverOptions.x0[1 + (2-1) * columns[1].ConvDispOpInstance.nPoints : 2 * columns[1].ConvDispOpInstance.nPoints] solverOptions.x0[1 + (3-1) * columns[1].ConvDispOpInstance.nPoints : 3 * columns[1].ConvDispOpInstance.nPoints] solverOptions.x0[1 + (4-1) * columns[1].ConvDispOpInstance.nPoints : 4 * columns[1].ConvDispOpInstance.nPoints]]' ./c_scale_data[2]
    c0 = solverOptions.x0[1 : 4*columns[1].ConvDispOpInstance.nPoints] # mobile phase
    q1 = init_conditions_function(input_init, NN_model, st, 1)
    q2 = init_conditions_function(input_init, NN_model, st, 2)
    q3 = init_conditions_function(input_init, NN_model, st, 3)
    q4 = init_conditions_function(input_init, NN_model, st, 4)
    solverOptions.x0 = vcat(c0, q1, q2, q3, q4)
    model_setup = hybrid_model(columns, columns[1].RHS_q, columns[1].cpp, columns[1].qq, 1, solverOptions.nColumns, solverOptions.idx_units, switches, solverOptions, c_scale_data[2], 1.0)
    solve_model_hybrid(columns=columns, switches=switches, solverOptions=solverOptions, hybrid_model_setup=model_setup, p_NN=NN_model, outlets=outlets, alg=FBDF(autodiff=false))

    ## plots 

    # plotting the training data
    p1 = plot()
    for j = 1:columns[1].nComp
        scatter!(p1, sol_times[1:test_split], data_full[p][1:test_split,j], label = components[j], seriescolor = colors[j], markersize = 5)
        plot!(p1, columns[1].solution_times[1:test_split_refined], columns[1].solution_outlet[1:test_split_refined,j], label = "HM$structure_best_plot - " * components[j], linewidth = 4, seriescolor = colors[j])
        plot!(p1, sol_times_mm[1:test_split_refined], columns_mm[1].solution_outlet[1:test_split_refined,j], label = "MM - " * components[j], linewidth = 4, seriescolor = colors_mm[j])
    end
    plot!(p1, legendfontsize = 16, labelfontsize = 16, tickfontsize = 16, legend = :none)
    if p==1
        plot!(p1, legendfontsize = 16, labelfontsize = 16, tickfontsize = 16, legend = :outerright)
    end
    xlabel!("Time (min)", guidefontsize = 16, tickfontsize = 16)
    ylabel!("Concentration / mol/L", guidefontsize = 16, tickfontsize = 16)
    display(p1)
    savefig(p1, joinpath(@__DIR__, "train_$p.svg"))


    # ploting the test data of the binary components
    p1 = plot()
    for j = 1:columns[1].nComp
        scatter!(p1, sol_times[test_split:end], data_full[p][test_split:end,j], label = components[j], seriescolor = colors[j], markersize = 5)
        plot!(p1, columns[1].solution_times[test_split_refined:end], columns[1].solution_outlet[test_split_refined:end,j], label = "HM$structure_best_plot - " * components[j], linewidth = 4, seriescolor = colors[j])
        plot!(p1, sol_times_mm[test_split_refined:end], columns_mm[1].solution_outlet[test_split_refined:end,j], label = "MM - " * components[j], linewidth = 4, seriescolor = colors_mm[j])
        
    end
    plot!(p1, legendfontsize = 16, labelfontsize = 16, tickfontsize = 16, legend = :none)
    if p==1
        plot!(p1, legendfontsize = 12, labelfontsize = 16, tickfontsize = 16, legend = :right)
        plot!(p1, ylims = (2.2, 6.3))
    end
    xlabel!("Time (min)", guidefontsize = 16, tickfontsize = 16)
    ylabel!("Concentration / mol/L", guidefontsize = 16, tickfontsize = 16)
    display(p1)
    savefig(p1, joinpath(@__DIR__, "test_$p.svg"))


    # plotting the training and testing data of the binary components
    p1 = plot()
    for j = 1:columns[1].nComp
        scatter!(p1, sol_times, data_full[p][:,j], label = components[j], seriescolor = colors[j], markersize = 5)
        plot!(p1, columns[1].solution_times, columns[1].solution_outlet[:,j], label = "HM$structure_best_plot - " * components[j], linewidth = 4, seriescolor = colors[j])
        plot!(p1, sol_times_mm, columns_mm[1].solution_outlet[:,j], label = "MM - " * components[j], linewidth = 4, seriescolor = colors_mm[j])
        
    end
    plot!(p1, legendfontsize = 16, labelfontsize = 16, tickfontsize = 16, legend = :none)
    # plot vertical line to indicate training/test split 
    plot!(p1, [data_trains[p][end,end], data_trains[p][end,end]], [0, maximum(columns[1].solution_outlet[:,:])], color = "black", linestyle =:dash, linewidth = 4)
    xlabel!("Time / min", guidefontsize = 16, tickfontsize = 16)
    ylabel!("Concentration / mol/L")
    display(p1)
    savefig(p1, joinpath(@__DIR__, "train_test_$p.svg"))

end


# Plotting the reactive data 
for p=1:4 
    if p==1
        dataset = "fig3a"
        data = c_exp_data3a
    elseif p==2
        dataset = "fig3b"
        data = c_exp_data3b
    elseif p==3
        dataset = "fig4a"
        data = c_exp_data4a
    elseif p==4
        dataset = "fig4b"
        data = c_exp_data4b
    end

    # set up times 
    sol_times = data[:,end]
    sol_times_mm = sort(union(collect(0:0.1:sol_times[end]), sol_times))

    # solve mechanistic model
    inlets, outlets, columns_mm, switches, solverOptions = model(dataset = dataset, sol_times = sol_times_mm)
    rhs_test = hybrid_model_mm(columns_mm, columns_mm[1].RHS_q, columns_mm[1].cpp, columns_mm[1].qq, 1, solverOptions.nColumns, solverOptions.idx_units, switches, solverOptions, c_scale_data[2], 1.0)
    solve_model_hybrid(columns=columns_mm, switches=switches, solverOptions=solverOptions, hybrid_model_setup=rhs_test, p_NN=NN_model_mm, outlets=outlets, alg=FBDF(autodiff=false))

    # solve hybrid model
    inlets, outlets, columns, switches, solverOptions = model(dataset = dataset, sol_times=sol_times_mm)
    input_init = [solverOptions.x0[1 + (1-1) * columns[1].ConvDispOpInstance.nPoints : 1 * columns[1].ConvDispOpInstance.nPoints] solverOptions.x0[1 + (2-1) * columns[1].ConvDispOpInstance.nPoints : 2 * columns[1].ConvDispOpInstance.nPoints] solverOptions.x0[1 + (3-1) * columns[1].ConvDispOpInstance.nPoints : 3 * columns[1].ConvDispOpInstance.nPoints] solverOptions.x0[1 + (4-1) * columns[1].ConvDispOpInstance.nPoints : 4 * columns[1].ConvDispOpInstance.nPoints]]' ./c_scale_data[2]
    c0 = solverOptions.x0[1 : 4*columns[1].ConvDispOpInstance.nPoints] # mobile phase
    q1 = init_conditions_function(input_init, NN_model, st, 1)
    q2 = init_conditions_function(input_init, NN_model, st, 2)
    q3 = init_conditions_function(input_init, NN_model, st, 3)
    q4 = init_conditions_function(input_init, NN_model, st, 4)
    solverOptions.x0 = vcat(c0, q1, q2, q3, q4)
    rhs_test = hybrid_model(columns, columns[1].RHS_q, columns[1].cpp, columns[1].qq, 1, solverOptions.nColumns, solverOptions.idx_units, switches, solverOptions, c_scale_data[2], 1.0)

    # If the hybrid model structure has neural network prediction stuff inside the reaction term, the reaction term should be updated with the neural network prediction.
    if structure_best == 4
    rhs_test = hybrid_model_reac(columns, columns[1].RHS_q, columns[1].cpp, columns[1].qq, 1, solverOptions.nColumns, solverOptions.idx_units, switches, solverOptions, c_scale_data[2], 1.0)
    columns[1].reaction_solid = reaction_HM4(
        k1 = columns[1].reaction_solid.k1,
        Keq = columns[1].reaction_solid.Keq,
        K = columns[1].reaction_solid.K,
        stoich = columns[1].reaction_solid.stoich,
        density = columns[1].reaction_solid.density,
        Vm = columns[1].reaction_solid.Vm,
        Tmodel = columns[1].reaction_solid.Tmodel,
        nComp = columns[1].nComp,
        compStride = columns[1].bindStride,
        T = columns[1].reaction_solid.T,
        nn = nn,
        nn_par = NN_model,
        st = st
        )
    elseif structure_best == 12
        rhs_test = hybrid_model_reac(columns, columns[1].RHS_q, columns[1].cpp, columns[1].qq, 1, solverOptions.nColumns, solverOptions.idx_units, switches, solverOptions, c_scale_data[2], 1.0)
        columns[1].reaction_solid = reaction_HM12(
            k1 = columns[1].reaction_solid.k1,
            Keq = columns[1].reaction_solid.Keq,
            K = columns[1].reaction_solid.K,
            stoich = columns[1].reaction_solid.stoich,
            density = columns[1].reaction_solid.density,
            Vm = columns[1].reaction_solid.Vm,
            Tmodel = columns[1].reaction_solid.Tmodel,
            nComp = columns[1].nComp,
            compStride = columns[1].bindStride,
            T = columns[1].reaction_solid.T,
            nn = nn,
            nn_par = NN_model,
            st = st
            )
    elseif structure_best == 15 || structure_best == 13
        rhs_test.columns[1].reaction_solid.K = NN_model.Keq # determined simultaneously w. weights/bias of NN
    end        
    solve_model_hybrid(columns=columns, switches=switches, solverOptions=solverOptions, hybrid_model_setup=rhs_test, p_NN=NN_model, outlets=outlets, alg=FBDF(autodiff=false))



    # plotting the reactive data
    p1 = plot()
    for j = 1:columns[1].nComp
        scatter!(p1, sol_times, data[:,j], label = components[j], seriescolor = colors[j], markersize = 5)
        plot!(p1, columns[1].solution_times, columns[1].solution_outlet[:,j], label = "HM$structure_best_plot - " * components[j], linewidth = 4, seriescolor = colors[j])
        plot!(p1, sol_times_mm, columns_mm[1].solution_outlet[:,j], label = "MM - " * components[j], linewidth = 4, seriescolor = colors_mm[j])
        
    end
    plot!(p1, legendfontsize = 11, labelfontsize = 16, tickfontsize = 16, legend = :none)
    if p==2
        plot!(p1, legend = :right)
    end
    # plot vertical line to indicate training/test split 
    xlabel!("Time / min", guidefontsize = 16, tickfontsize = 16)
    ylabel!("Concentration / mol/L", guidefontsize = 16, tickfontsize = 16)
    display(p1)
    savefig(p1, joinpath(@__DIR__, "test_$dataset.svg"))


end







# plotting the NN predictions raw
Nsamples = 100
c1 = LinRange(0,13.4, Nsamples) # collect(0:0.1:maximum(c_exp_data[:, 1]))
c2 = LinRange(0,13.4, Nsamples) # collect(0:0.1:maximum(c_exp_data[:, 1]))
grid = [ (x, y) for x in c1, y in c2 ]
grid_matrix = hcat([collect(x) for x in grid]...)
grid_matrix = vcat(grid_matrix, zeros(2, size(grid_matrix, 2)))
nn_predictions = nn(grid_matrix./c_scale_data[2], NN_model.p1, st)[1][1:2, 1:end]

for i in 1:2
    p1 = plot(layout=(1, 1), size=(650, 500), margin=10Plots.mm)
    plot!(p1, tickfontsize = 17, guidefontsize = 26, legend = :none)
    surface!(p1, c2, c1, reshape(nn_predictions[i,:].*50, length(c2), length(c1)), dpi=600)
    xlabel!(L"c_{\textrm{AcidPro}} ~ / ~ \textrm{mol/L}") # the axis are switched when plotted!
    ylabel!(L"c_{\textrm{POH}} ~ / ~ \textrm{mol/L}")
    if i==1
        zlabel!(L"H_{\textrm{POH}}")
    elseif i==2
        zlabel!(L"H_{\textrm{AcidPro}}")
    end
    display(p1)
    savefig(p1, joinpath(@__DIR__, "Predictions_$i.svg"))
end

# plot(c1./c_scale_data[2], nn([c1./c_scale_data[2] zeros(100) zeros(100) zeros(100)]', NN_model.p1, st)[1][1, 1:end])
# plot!(c2./c_scale_data[2], nn([zeros(100) c2./c_scale_data[2] zeros(100) zeros(100)]', NN_model.p1, st)[1][2, 1:end])

# plotting the q predictions to check for consistency
for i in 1:2
    p1 = plot(layout=(1, 1), size=(650, 500), margin=10Plots.mm)
    plot!(p1, tickfontsize = 17, guidefontsize = 26, legend = :none)
    q_predictions = init_conditions_function(grid_matrix./c_scale_data[2], NN_model, st, i)
    surface!(p1, c2, c1, reshape(q_predictions, length(c2), length(c1)), dpi=600)
    xlabel!(L"c_{\textrm{AcidPro}} ~ / ~ \textrm{mol/L}") # the axis are switched when plotted!
    ylabel!(L"c_{\textrm{POH}} ~ / ~ \textrm{mol/L}")
    if i==1
        zlabel!(L"q_{\textrm{POH}}")
    elseif i==2
        zlabel!(L"q_{\textrm{AcidPro}}")
    end
    display(p1)
    savefig(p1, joinpath(@__DIR__, "Predictions_$(i)_q.svg"))
end



# Read the convergence data and plot the convergence data. 
df1 = CSV.read(joinpath(@__DIR__, "..", "model_structure_4", "neurons_5", "convergence.csv"), DataFrame) 
df2 = CSV.read(joinpath(@__DIR__, "..", "model_structure_8", "neurons_3_2", "convergence_2.csv"), DataFrame) 
df3 = CSV.read(joinpath(@__DIR__, "..", "model_structure_15", "neurons_3_2", "convergence_2.csv"), DataFrame) 
df = (df1, df2, df3)
labels = ["HM4", "HM7", "HM10"]
p1 = plot()
for i=1:3 #i=1

    # Find the last 0 in the data 
    idxx = findlast(x -> x == 1, df[i][:, 1])
    plot!(p1, df[i][idxx:end-1, 2], label = labels[i], linewidth = 4)
    xlabel!("Epochs")
    ylabel!("Loss")
end
plot!(p1, legendfontsize = 16, labelfontsize = 16, tickfontsize = 16, legend = :right)
display(p1)
savefig(p1, joinpath(@__DIR__, "convergences.svg"))


