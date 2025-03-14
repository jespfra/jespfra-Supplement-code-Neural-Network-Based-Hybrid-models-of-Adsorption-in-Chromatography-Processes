

# Add the include file custom to load packages and scripts. 
# the file is located on the main from which the file takes care of the rest. 
include(joinpath(@__DIR__,"..","..","..","..","..","..","include.jl"))
include(joinpath(@__DIR__,"..","..","..","..","include_hybrid.jl"))


############## Solve using a hybrid modelling approach ##############
include(joinpath(@__DIR__,"..", "..", "loss_functions.jl"))
include(joinpath(@__DIR__, "hybrid_model_setup.jl"))
include(joinpath(@__DIR__,"..", "..", "lrm_model.jl"))
# include(joinpath(@__DIR__,".." ,"plot_functions.jl"))
include(joinpath(@__DIR__,"..", "..", "get_data.jl"))
include(joinpath(@__DIR__,"..", "..", "get_initial_conditions.jl"))
include(joinpath(@__DIR__,"..", "..", "evaluate_metrics.jl"))
using LaTeXStrings


# Get data 
test_split = 204
c_exp_data, qc_exp_data, input_pretrain, output_pretrain, data_train_c, data_train_q, data_test_c, data_test_q, c_scale_data, q_scale_data = get_data(test_split=test_split)


## evaluation of the mechanistic model and the best performing hybrid model. 

# first plotting the mechanistic model
# solving the mechanistic model 
ka = [11.66, 5.08]
qmax = [9.13, 5.11]
kl = [0.51, 0.51]

NN_model = ComponentArray(ka = ka, qmax = qmax, kl = kl)
sol_times = sort(union(collect(0:0.1:174.4), [0.0, 27.287, 57.387, 86.267, 116.667, 147.067, 174.427]))
test_split_refined = findfirst(x -> x == 147.067, sol_times)
inlets, outlets, columns_mm, switches, solverOptions = model(sol_times=sol_times)
model_setup = hybrid_model_mm(columns_mm, columns_mm[1].RHS_q, columns_mm[1].cpp, columns_mm[1].qq, 1, solverOptions.nColumns, solverOptions.idx_units, switches, c_scale_data, q_scale_data)
solve_model_hybrid(columns=columns_mm, switches=switches, solverOptions=solverOptions, hybrid_model_setup=model_setup, p_NN=NN_model, outlets=outlets, alg=FBDF(autodiff=false))

# solving the best performing hybrid model
structure_best = 14
structure_best_plot = 8

include(joinpath(@__DIR__, "..", "model_structure_$(structure_best)", "hybrid_model_setup.jl"))
df_avg = CSV.read(joinpath(@__DIR__,  "..", "model_structure_$(structure_best)", "metrics_train_avg_2.csv"), DataFrame)
idx_min = findfirst(x -> x == minimum(df_avg.MSE_test), df_avg.MSE_test)
NN_model = load(joinpath(@__DIR__, "..", "model_structure_$(structure_best)", "neurons_$(df_avg.neurons[idx_min])_2","NN_model.jld2"))["NN_model"]
nn, p_init, st = initialize_neural_network(saveat = joinpath(@__DIR__), input = 2, neurons = idx_min, output = 2, activation = (tanh_fast, softplus))

inlets, outlets, columns, switches, solverOptions = model(sol_times=sol_times)
input_init = [solverOptions.x0[1:columns[1].ConvDispOpInstance.nPoints] solverOptions.x0[1+columns[1].ConvDispOpInstance.nPoints:2*columns[1].ConvDispOpInstance.nPoints]]' ./c_scale_data
c0 = solverOptions.x0[1:2*columns[1].ConvDispOpInstance.nPoints]
q1 = init_conditions_function(input_init, NN_model, st, 1)
q2 = init_conditions_function(input_init, NN_model, st, 2)
solverOptions.x0 = vcat(c0, q1, q2)
model_setup = hybrid_model(columns, columns[1].RHS_q, columns[1].cpp, columns[1].qq, 1, solverOptions.nColumns, solverOptions.idx_units, switches, c_scale_data, q_scale_data)
solve_model_hybrid(columns=columns, switches=switches, solverOptions=solverOptions, hybrid_model_setup=model_setup, p_NN=NN_model, outlets=outlets, alg=FBDF(autodiff=false))


# plotting the model performances - training
colors = ["blue", "red"]
colors_mm = ["magenta", "olive"]
components = ["POH", "ProPro"]
p1 = plot() 
for i in 1:2
    scatter!(p1, c_exp_data[1:test_split, end], data_train_c[i]*c_scale_data, label = components[i], seriescolor = colors[i], markersize = 5)
    plot!(p1, columns[1].solution_times[1:test_split_refined], columns[1].solution_outlet[1:test_split_refined,i], label="HM$structure_best_plot - " * components[i], seriescolor = colors[i], linewidth = 4)
    plot!(p1, columns_mm[1].solution_times[1:test_split_refined], columns_mm[1].solution_outlet[1:test_split_refined,i], label="MM - " * components[i], seriescolor = colors_mm[i], linewidth = 4)
    
end
plot!(p1, legendfontsize = 16, labelfontsize = 16, tickfontsize = 16)
xlabel!("Time / min")
ylabel!("Concentration / mol/L")
display(p1)
savefig(p1, joinpath(@__DIR__, "train.svg"))


# plotting the model performances - test
p1 = plot() 
for i in 1:2
    scatter!(p1, c_exp_data[test_split:end, end], data_test_c[i]*c_scale_data, label = components[i], seriescolor = colors[i], markersize = 5)
    plot!(p1, columns[1].solution_times[test_split_refined:end], columns[1].solution_outlet[test_split_refined:end, i], label="HM$structure_best_plot - " * components[i], seriescolor = colors[i], linewidth = 4)
    plot!(p1, columns_mm[1].solution_times[test_split_refined:end], columns_mm[1].solution_outlet[test_split_refined:end, i], label="MM - " * components[i], seriescolor = colors_mm[i], linewidth = 4)
    
end
plot!(p1, legendfontsize = 16, labelfontsize = 16, tickfontsize = 16, legend = :right)
xlabel!("Time / min")
ylabel!("Concentration / mol/L")
display(p1)
savefig(p1, joinpath(@__DIR__, "test.svg"))


# plotting the model performances - training and testing
p1 = plot() 
for i in 1:2
    scatter!(p1, c_exp_data[:, end], c_exp_data[:,i], label = components[i], seriescolor = colors[i], markersize = 5)
    plot!(p1, columns[1].solution_times, columns[1].solution_outlet[:,i], label="HM$structure_best_plot - " * components[i], seriescolor = colors[i], linewidth = 4)
    plot!(p1, columns_mm[1].solution_times, columns_mm[1].solution_outlet[:,i], label="MM - " * components[i], seriescolor = colors_mm[i], linewidth = 4)
    
end
plot!(p1, [outlets[1].solution_times[test_split_refined], outlets[1].solution_times[test_split_refined]], [0, maximum(outlets[1].solution_outlet[:,:])], color = "black", linestyle =:dash, linewidth = 4)
plot!(p1, legendfontsize = 16, labelfontsize = 16, tickfontsize = 16, legend = :none)
xlabel!("Time / min")
ylabel!("Concentration / mol/L")
display(p1)
savefig(p1, joinpath(@__DIR__, "train_test.svg"))


# plotting the NN predictions raw
Nsamples = 100
c1 = LinRange(0,maximum(c_exp_data[:, 1])+0.2, Nsamples) # collect(0:0.1:maximum(c_exp_data[:, 1]))
c2 = LinRange(0,maximum(c_exp_data[:, 2])+0.2, Nsamples) # collect(0:0.1:maximum(c_exp_data[:, 1]))
grid = [ (x, y) for x in c1, y in c2 ]
grid_matrix = hcat([collect(x) for x in grid]...)
nn_predictions = nn(grid_matrix./c_scale_data, NN_model.p1, st)[1][1:2, 1:end]

for i in 1:2 #i=2
    p1 = plot(layout=(1, 1), size=(650, 500), margin=10Plots.mm)
    plot!(p1, legendfontsize = 14, labelfontsize = 26, tickfontsize = 14, legend = :none)


    # Training and test predictions 
    z_train = nn([data_train_c[1] data_train_c[2]]', NN_model.p1, st)[1][1:2, 1:end]
    z_test = nn([data_test_c[1] data_test_c[2]]', NN_model.p1, st)[1][1:2, 1:end]
    
    if i==1
        zlabel!(L"K_{\textrm{POH}}")
        # surface!(p1, c1, c2, reshape(nn_predictions[i,:], length(c2), length(c1))', dpi=600) #because it is transposed, the x and y axis are switched!
        surface!(p1, c1, c2, reshape(nn_predictions[i,:].*7.5, length(c1), length(c2))', dpi=600) 
        xlabel!(L"c_{\textrm{POH}} ~ / ~ \textrm{mol/L}")
        ylabel!(L"c_{\textrm{ProPro}} ~ / ~ \textrm{mol/L}")

        # Plotting the training data and predictions as points 
        scatter!(p1, data_train_c[1].*c_scale_data, data_train_c[2].*c_scale_data, z_train[i,:].*7.5, zcolor=z_train[i,:], label = "Training data", markersize = 6) # x-y axes are reverted, this is the proof!
        
        # Plotting the test data and predictions as points with circles with pink color
        scatter!(p1, data_test_c[1].*c_scale_data, data_test_c[2].*c_scale_data, z_test[i,:].*7.5, zcolor=z_test[i,:], label = "Test data", markersize = 10, markerstrokecolor = :pink) # x-y axes are reverted, this is the proof!

    elseif i==2
        zlabel!(L"K_{\textrm{ProPro}}")
        # surface!(p1, c1, c2, reshape(nn_predictions[i,:], length(c2), length(c1))', dpi=600) #because it is transposed, the x and y axis are switched!
        surface!(p1, c1, c2, reshape(nn_predictions[i,:].*7.5, length(c1), length(c2))', dpi=600) 
        xlabel!(L"c_{\textrm{POH}} ~ / ~ \textrm{mol/L}")
        ylabel!(L"c_{\textrm{ProPro}} ~ / ~ \textrm{mol/L}")

        # Plotting the training data and predictions as points 
        scatter!(p1, data_train_c[1].*c_scale_data, data_train_c[2].*c_scale_data, z_train[i,:].*7.5, zcolor=z_train[i,:], label = "Training data", markersize = 6) # x-y axes are reverted, this is the proof!
        
        # Plotting the test data and predictions as points with circles with pink color
        scatter!(p1, data_test_c[1].*c_scale_data, data_test_c[2].*c_scale_data, z_test[i,:].*7.5, zcolor=z_test[i,:], label = "Test data", markersize = 10, markerstrokecolor = :pink) # x-y axes are reverted, this is the proof!
    end

    display(p1)
    savefig(p1, joinpath(@__DIR__, "Predictions_$i.svg"))
end




# plotting the q predictions to check for consistency

for i in 1:2 #i=2
    p1 = plot(layout=(1, 1), size=(650, 500), margin=10Plots.mm)
    plot!(p1, legendfontsize = 14, labelfontsize = 26, tickfontsize = 14, legend = :none)

    q_predictions = init_conditions_function(grid_matrix./c_scale_data, NN_model, st, i)
    # Training and test predictions 
    # z_train = nn([data_train_c[1] data_train_c[2]]', NN_model.p1, st)[1][1:2, 1:end]
    # z_test = nn([data_test_c[1] data_test_c[2]]', NN_model.p1, st)[1][1:2, 1:end]
    
    if i==1
        zlabel!(L"q_{\textrm{POH}}")
        # surface!(p1, c1, c2, reshape(nn_predictions[i,:], length(c2), length(c1))', dpi=600) #because it is transposed, the x and y axis are switched!
        surface!(p1, c1, c2, reshape(q_predictions, length(c1), length(c2))', dpi=600) 
        xlabel!(L"c_{\textrm{POH}} ~ / ~ \textrm{mol/L}")
        ylabel!(L"c_{\textrm{ProPro}} ~ / ~ \textrm{mol/L}")

        # Plotting the training data and predictions as points 
        # scatter!(p1, data_train_c[1].*c_scale_data, data_train_c[2].*c_scale_data, z_train[i,:].*7.5, zcolor=z_train[i,:], label = "Training data", markersize = 6) # x-y axes are reverted, this is the proof!
        
        # Plotting the test data and predictions as points with circles with pink color
        # scatter!(p1, data_test_c[1].*c_scale_data, data_test_c[2].*c_scale_data, z_test[i,:].*7.5, zcolor=z_test[i,:], label = "Test data", markersize = 10, markerstrokecolor = :pink) # x-y axes are reverted, this is the proof!

    elseif i==2
        zlabel!(L"q_{\textrm{ProPro}}")
        # surface!(p1, c1, c2, reshape(nn_predictions[i,:], length(c2), length(c1))', dpi=600) #because it is transposed, the x and y axis are switched!
        surface!(p1, c1, c2, reshape(q_predictions, length(c1), length(c2))', dpi=600) 
        xlabel!(L"c_{\textrm{POH}} ~ / ~ \textrm{mol/L}")
        ylabel!(L"c_{\textrm{ProPro}} ~ / ~ \textrm{mol/L}")

        # Plotting the training data and predictions as points 
        # scatter!(p1, data_train_c[1].*c_scale_data, data_train_c[2].*c_scale_data, z_train[i,:].*7.5, zcolor=z_train[i,:], label = "Training data", markersize = 6) # x-y axes are reverted, this is the proof!
        
        # Plotting the test data and predictions as points with circles with pink color
        # scatter!(p1, data_test_c[1].*c_scale_data, data_test_c[2].*c_scale_data, z_test[i,:].*7.5, zcolor=z_test[i,:], label = "Test data", markersize = 10, markerstrokecolor = :pink) # x-y axes are reverted, this is the proof!
    end

    display(p1)
    savefig(p1, joinpath(@__DIR__, "Predictions_$(i)_q.svg"))
end



# Read the convergence data and plot the convergence data. 
df1 = CSV.read(joinpath(@__DIR__, "..", "model_structure_1", "neurons_7", "convergence.csv"), DataFrame) #
df2 = CSV.read(joinpath(@__DIR__, "..", "model_structure_4", "neurons_9", "convergence.csv"), DataFrame) # 
df3 = CSV.read(joinpath(@__DIR__, "..", "model_structure_14", "neurons_8_2", "convergence_2.csv"), DataFrame) # Correponds to model 6, using 7 neurons a 1 layer 
df = (df1, df2, df3)
labels = ["HM2", "HM5", "HM8"]
p1 = plot()
for i=1:3 #i=1

    # Find the last 0 in the data 
    idxx = findlast(x -> x == 1, df[i][:, 1])
    plot!(p1, df[i][idxx:end-1, 2], label = labels[i], linewidth = 4)
    xlabel!("Epochs")
    ylabel!("Loss")
end
ylims!(1e-6, maximum(df2[:,2])+1e-6)
plot!(p1, legendfontsize = 16, labelfontsize = 16, tickfontsize = 16) #legend = (0.85, 0.3)
display(p1)
savefig(p1, joinpath(@__DIR__, "convergences.svg"))