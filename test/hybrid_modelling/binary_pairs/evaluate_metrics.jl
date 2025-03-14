

function evaluate_metrics_training(saveat, l, layers = 1, HM = 1)
    """
    A scipt to evaluate the metrics for validation data and plot it. 
	l is number of neurons 
    layers is the number of layers in the NN. 

    """
    # For plotting pretrain 
    if HM != 0
        nnmodel = u0 
        pretrain = "_pretrain"
        evaluate_plots(saveat, l, layers, HM, nnmodel, pretrain)
    end

    # For plotting trained 
    nnmodel = NN_model 
    pretrain = ""
    evaluate_plots(saveat, l, layers, HM, nnmodel, pretrain)
    
end

function evaluate_plots(saveat, l, layers, HM, nnmodel, pretrain)
    colors = ["blue", "green", "purple", "red"]
    diff = Float64[]; ndiff = Float64[]; loss = 0.0
    for p in selection #p=1
		
        sol_times = data_trains[p][:,end]
        inlets, outlets, columns, switches, solverOptions = model(dataset = "$p", sol_times = sol_times)
        if HM != 0
            input_init = [solverOptions.x0[1 + (1-1) * columns[1].ConvDispOpInstance.nPoints : 1 * columns[1].ConvDispOpInstance.nPoints] solverOptions.x0[1 + (2-1) * columns[1].ConvDispOpInstance.nPoints : 2 * columns[1].ConvDispOpInstance.nPoints] solverOptions.x0[1 + (3-1) * columns[1].ConvDispOpInstance.nPoints : 3 * columns[1].ConvDispOpInstance.nPoints] solverOptions.x0[1 + (4-1) * columns[1].ConvDispOpInstance.nPoints : 4 * columns[1].ConvDispOpInstance.nPoints]]' ./c_scale_data[2]
            c0 = solverOptions.x0[1 : 4*columns[1].ConvDispOpInstance.nPoints] # mobile phase
            q1 = init_conditions_function(input_init, nnmodel, st, 1)
            q2 = init_conditions_function(input_init, nnmodel, st, 2)
            q3 = init_conditions_function(input_init, nnmodel, st, 3)
            q4 = init_conditions_function(input_init, nnmodel, st, 4)
            solverOptions.x0 = vcat(c0, q1, q2, q3, q4)
        end
        rhs_test = hybrid_model(columns, columns[1].RHS_q, columns[1].cpp, columns[1].qq, 1, solverOptions.nColumns, solverOptions.idx_units, switches, solverOptions, c_scale_data[2], 1.0)
		solve_model_hybrid(columns=columns, switches=switches, solverOptions=solverOptions, hybrid_model_setup=rhs_test, p_NN=nnmodel, outlets=outlets, alg=FBDF(autodiff=false))

		# Determine metrics and plot
        p1 = plot()
		for j = 1:columns[1].nComp
			diff1 =  abs.(data_trains[p][:,j] .*c_scale_data[p] .-  columns[1].solution_outlet[:,j])
			ndiff1 =  abs.(data_trains[p][:,j] .-  columns[1].solution_outlet[:,j]./c_scale_data[p])
            diff1 = skipmissing(diff1)
			ndiff1 = skipmissing(ndiff1)
            append!(diff, diff1)
            append!(ndiff, ndiff1)
            loss += 1/length(data_trains[p][:,j])^2 * sum(ndiff1)^2

            # plots 
            scatter!(p1, sol_times, data_trains[p][:,j]*c_scale_data[p], label = "Experimental - C$j", seriescolor = colors[j])
            plot!(p1, sol_times, columns[1].solution_outlet[:,j], label = "Model - C$j", linewidth = 3, seriescolor = colors[j])
		end
        xlabel!(p1, "Time / min")
        ylabel!(p1, "Concentration / mol/L")
        plot!(p1, legendfontsize = 12, labelfontsize = 12, tickfontsize = 12)

        if layers==1
            savefig(p1,joinpath(saveat,"neurons_$l","plot_training_$(p)$pretrain.svg"))
        else
            savefig(p1,joinpath(saveat,"neurons_$(l)_$layers","plot_training_$(p)_$(layers)$pretrain.svg"))
        end

	end
    # display(p1)
	
    MAE = mean(diff)
    MSE = mean(diff.^2)
    NMSE = mean(ndiff.^2)
    RMSE = sqrt(mean(diff.^2))*100
    # Save DataFrame to a CSV file
	df = DataFrame(MAE_train = MAE, MSE_train = MSE, NMSE_train = NMSE, loss_train = loss, RMSE_train = RMSE)
    if layers==1
        CSV.write(joinpath(saveat, "neurons_$l","metrics_train$pretrain.csv"), df)
    else
        CSV.write(joinpath(saveat, "neurons_$(l)_$layers","metrics_train_$(layers)$pretrain.csv"), df)
    end


    # For full 'training' data evaluation 
    diff = Float64[]; ndiff = Float64[]; loss = 0.0
    for p in selection #
        # println(p)
        sol_times = data_full[p][:,end]
        inlets, outlets, columns, switches, solverOptions = model(; dataset = "$p", sol_times = sol_times)
        if HM != 0
            input_init = [solverOptions.x0[1 + (1-1) * columns[1].ConvDispOpInstance.nPoints : 1 * columns[1].ConvDispOpInstance.nPoints] solverOptions.x0[1 + (2-1) * columns[1].ConvDispOpInstance.nPoints : 2 * columns[1].ConvDispOpInstance.nPoints] solverOptions.x0[1 + (3-1) * columns[1].ConvDispOpInstance.nPoints : 3 * columns[1].ConvDispOpInstance.nPoints] solverOptions.x0[1 + (4-1) * columns[1].ConvDispOpInstance.nPoints : 4 * columns[1].ConvDispOpInstance.nPoints]]' ./c_scale_data[2]
            c0 = solverOptions.x0[1 : 4*columns[1].ConvDispOpInstance.nPoints] # mobile phase
            q1 = init_conditions_function(input_init, nnmodel, st, 1)
            q2 = init_conditions_function(input_init, nnmodel, st, 2)
            q3 = init_conditions_function(input_init, nnmodel, st, 3)
            q4 = init_conditions_function(input_init, nnmodel, st, 4)
            solverOptions.x0 = vcat(c0, q1, q2, q3, q4)
        end
        rhs_test = hybrid_model(columns, columns[1].RHS_q, columns[1].cpp, columns[1].qq, 1, solverOptions.nColumns, solverOptions.idx_units, switches, solverOptions, c_scale_data[2], 1.0)
        solve_model_hybrid(columns=columns, switches=switches, solverOptions=solverOptions, hybrid_model_setup=rhs_test, p_NN=nnmodel, outlets=outlets, alg=FBDF(autodiff=false))

        # Determine metrics and plot
        p1 = plot()
        for j = 1:columns[1].nComp
            diff1 =  abs.(data_full[p][:,j] .-  columns[1].solution_outlet[:,j])
			ndiff1 =  abs.(data_full[p][:,j] ./c_scale_data[p] .-  columns[1].solution_outlet[:,j] ./c_scale_data[p])
            diff1 = skipmissing(diff1)
			ndiff1 = skipmissing(ndiff1)
            append!(diff, diff1)
            append!(ndiff, ndiff1)
            loss += 1/length(data_full[p][:,j])^2 * sum(ndiff1)^2

            # plots 
            scatter!(p1, sol_times, data_full[p][:,j], label = "Experimental - C$j", seriescolor = colors[j])
            plot!(p1, sol_times, columns[1].solution_outlet[:,j], label = "Model - C$j", seriescolor = colors[j], linewidth = 3)
        end
        # plot vertical line to indicate training/test split 
        plot!(p1, [data_trains[p][end,end], data_trains[p][end,end]], [0, maximum(columns[1].solution_outlet[:,:])], color = "black", linestyle =:dash, linewidth = 4)

        xlabel!(p1, "Time / min")
        ylabel!(p1, "Concentration / mol/L")
        # plot!(p1, legendfontsize = 12, labelfontsize = 12, tickfontsize = 12, legendalpha=0.5)
        plot!(p1, legend=:none)

        if layers==1
            savefig(p1,joinpath(saveat,"neurons_$l","plot_training_full_$(p)$pretrain.svg"))
        else
            savefig(p1,joinpath(saveat,"neurons_$(l)_$layers","plot_training_full_$(p)_$(layers)$pretrain.svg"))
        end
    end

    # Save DataFrame to a CSV file
    MAE = mean(diff)
    MSE = mean(diff.^2)
    NMSE = mean(ndiff.^2)
    RMSE = sqrt(mean(diff.^2))*100

    MAE_test = MAE - df[!,"MAE_train"][1]; MSE_test = MSE - df[!,"MSE_train"][1]; NMSE_test = NMSE -  df[!,"NMSE_train"][1]; loss_test = loss - df[!,"loss_train"][1]; RMSE_test = RMSE - df[!,"RMSE_train"][1];
    df = DataFrame(MAE = MAE_test, MSE = MSE_test, NMSE = NMSE_test, loss = loss_test, RMSE = RMSE_test)
    if layers==1
        CSV.write(joinpath(saveat,"neurons_$l","metrics_test_full$pretrain.csv"), df)
    else
        CSV.write(joinpath(saveat,"neurons_$(l)_$layers","metrics_test_full_$(layers)$pretrain.csv"), df)
    end

    # Evaluate test metrics
    diff = Float64[]; ndiff = Float64[]; loss = 0.0
    for i=1:4 
        if i==1
            dataset = "fig3a"
            data = c_exp_data3a
        elseif i==2
            dataset = "fig3b"
            data = c_exp_data3b
        elseif i==3
            dataset = "fig4a"
            data = c_exp_data4a
        elseif i==4
            dataset = "fig4b"
            data = c_exp_data4b
        end
        sol_times = data[:,end]
        inlets, outlets, columns, switches, solverOptions = model(dataset = dataset, sol_times = sol_times)
        # if hybrid model structure 4 is chosen, the reaction term should adapt the adsorption constants in the reaction term predicted by the neural network.
        rhs_test = hybrid_model(columns, columns[1].RHS_q, columns[1].cpp, columns[1].qq, 1, solverOptions.nColumns, solverOptions.idx_units, switches, solverOptions, c_scale_data[2], 1.0)

        # input for NNs to specify q0 according to NN predictions for reactive data
        if HM != 0
            input_init = [solverOptions.x0[1 + (1-1) * columns[1].ConvDispOpInstance.nPoints : 1 * columns[1].ConvDispOpInstance.nPoints] solverOptions.x0[1 + (2-1) * columns[1].ConvDispOpInstance.nPoints : 2 * columns[1].ConvDispOpInstance.nPoints] solverOptions.x0[1 + (3-1) * columns[1].ConvDispOpInstance.nPoints : 3 * columns[1].ConvDispOpInstance.nPoints] solverOptions.x0[1 + (4-1) * columns[1].ConvDispOpInstance.nPoints : 4 * columns[1].ConvDispOpInstance.nPoints]]' ./c_scale_data[2]
            c0 = solverOptions.x0[1 : 4*columns[1].ConvDispOpInstance.nPoints] # mobile phase
            q1 = init_conditions_function(input_init, nnmodel, st, 1)
            q2 = init_conditions_function(input_init, nnmodel, st, 2)
            q3 = init_conditions_function(input_init, nnmodel, st, 3)
            q4 = init_conditions_function(input_init, nnmodel, st, 4)
            solverOptions.x0 = vcat(c0, q1, q2, q3, q4)
        end
        
        if HM == 4 || HM == 8
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
                nn_par = nnmodel,
                st = st
                )

        elseif HM == 12 || HM == 14
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
                nn_par = nnmodel,
                st = st
                )

        elseif HM == 15 || HM == 13
            rhs_test.columns[1].reaction_solid.K = nnmodel.Keq # determined simultaneously w. weights/bias of NN

        end
        solve_model_hybrid(columns=columns, switches=switches, solverOptions=solverOptions, hybrid_model_setup=rhs_test, p_NN=nnmodel, outlets=outlets, alg=FBDF(autodiff=false))


        # Determine loss 
        fig = plot()
        for j = 1: columns[1].nComp #j=1
            diff1 =  abs.(data[:,j] .-  columns[1].solution_outlet[:,j])
			ndiff1 =  abs.(data[:,j] ./c_scale_data[3] .-  columns[1].solution_outlet[:,j] ./c_scale_data[3])
            diff1 = skipmissing(diff1)
			ndiff1 = skipmissing(ndiff1)
            append!(diff, diff1)
            append!(ndiff, ndiff1)
            loss += 1/length(data[:,j])^2 * sum(ndiff1)^2

            # plot 
            scatter!(fig, sol_times, data[:,j], label = "Experimental - C$j", seriescolor = colors[j], legend =:topright)
            plot!(fig, sol_times, columns[1].solution_outlet[:,j], label = "Model - C$j", seriescolor = colors[j], legend=:topright, linewidth = 3)
        end

        xlabel!("Time / min")
        ylabel!("Concentration / mol/L")
        plot!(fig, legendfontsize = 12, labelfontsize = 12, tickfontsize = 12)
        plot!(fig, legend=:none)
        # display(fig)
        if layers==1
            savefig(fig,joinpath(saveat, "neurons_$l","test_$(dataset)$pretrain.svg"))
        else
            savefig(fig,joinpath(saveat, "neurons_$(l)_$layers","test_$(dataset)_$(layers)$pretrain.svg"))
        end
	end

    # Save DataFrame to a CSV file
    MAE_test = mean(diff)
    MSE_test = mean(diff.^2)
    NMSE_test = mean(ndiff.^2)
    RMSE_test = sqrt(mean(diff.^2))*100

    df = DataFrame(MAE = MAE_test, MSE = MSE_test, NMSE = NMSE_test, loss = loss_test, RMSE = RMSE_test)
    if layers==1
        CSV.write(joinpath(saveat,"neurons_$l","metrics_test$pretrain.csv"), df)
    else
        CSV.write(joinpath(saveat,"neurons_$(l)_$layers","metrics_test_$(layers)$pretrain.csv"), df)
    end
    
    # plot convergence 
    if layers == 1
        convergence = CSV.read(joinpath(saveat,"neurons_$l","convergence.csv"), DataFrame)
    else
        convergence = CSV.read(joinpath(saveat,"neurons_$(l)_$layers","convergence_2.csv"), DataFrame)
    end 
    fig = plot()
    idx_convergence = findlast(x -> x == 1, convergence[:,1])
    scatter!(fig, convergence[idx_convergence:end,2], seriescolor = colors[1])
    xlabel!("Epochs")
    ylabel!("Loss")
    plot!(fig, legendfontsize = 12, labelfontsize = 12, tickfontsize = 12)
    plot!(fig, legend=:none)
    if layers==1
        savefig(fig,joinpath(saveat, "neurons_$l","convergence.svg"))
    else
        savefig(fig,joinpath(saveat, "neurons_$(l)_$layers","convergence_2.svg"))
    end
end



function evaluate_metrics_training_all(saveat, layers=1, HM = 1)
    """
    A scipt to evaluate the metrics for training data. 

    """
    # For plotting pretrain 
    if HM != 0
        pretrain = "_pretrain"
        evaluate_training(saveat, layers, pretrain)
    end

    # For plotting trained 
    pretrain = ""
    evaluate_training(saveat, layers, pretrain)
end
    

function evaluate_training(saveat, layers, pretrain) 

    df_total = DataFrame(neurons=neurons_test, 
                        MAE_training=zeros(length(neurons_test)), 
                        MSE_training=zeros(length(neurons_test)), 
                        NMSE_training=zeros(length(neurons_test)), 
                        loss_training=zeros(length(neurons_test)),
                        RMSE_training=zeros(length(neurons_test)),

                        MAE_test_binary=zeros(length(neurons_test)), 
                        MSE_test_binary=zeros(length(neurons_test)), 
                        NMSE_test_binary=zeros(length(neurons_test)), 
                        loss_test_binary=zeros(length(neurons_test)),
                        RMSE_test_binary=zeros(length(neurons_test)),
                        
                        MAE_test=zeros(length(neurons_test)), 
                        MSE_test=zeros(length(neurons_test)), 
                        NMSE_test=zeros(length(neurons_test)), 
                        loss_test=zeros(length(neurons_test)),
                        RMSE_test=zeros(length(neurons_test)))
    

                        #l=2
    if layers==1
        for l in neurons_test
            df1 = CSV.read(joinpath(saveat, "neurons_$l","metrics_train$pretrain.csv"), DataFrame)
            df2 = CSV.read(joinpath(saveat, "neurons_$l","metrics_test$pretrain.csv"), DataFrame)
            df3 = CSV.read(joinpath(saveat, "neurons_$l","metrics_test_full$pretrain.csv"), DataFrame)

            df_total[!,"MAE_training"][l-1] = mean([df1[!,"MAE_train"][1] ])#, df2[!,"MAE_train"][1], df3[!,"MAE_train"][1], df4[!,"MAE_train"][1], df5[!,"MAE_train"][1]])
            df_total[!,"MSE_training"][l-1] = mean([df1[!,"MSE_train"][1] ]) #, df2[!,"MSE_train"][1], df3[!,"MSE_train"][1], df4[!,"MSE_train"][1], df5[!,"MSE_train"][1]])
            df_total[!,"NMSE_training"][l-1] = mean([df1[!,"NMSE_train"][1] ]) 
            df_total[!,"loss_training"][l-1] = mean([df1[!,"loss_train"][1] ]) 
            df_total[!,"RMSE_training"][l-1] = mean([df1[!,"RMSE_train"][1] ]) #, df2[!,"RMSE_train"][1], df3[!,"RMSE_train"][1], df4[!,"RMSE_train"][1], df5[!,"RMSE_train"][1]])

            df_total[!,"MAE_test_binary"][l-1] = mean([df3[!,"MAE"][1] ])#, 
            df_total[!,"MSE_test_binary"][l-1] = mean([df3[!,"MSE"][1] ]) #,
            df_total[!,"NMSE_test_binary"][l-1] = mean([df3[!,"NMSE"][1] ])
            df_total[!,"loss_test_binary"][l-1] = mean([df3[!,"loss"][1] ])
            df_total[!,"RMSE_test_binary"][l-1] = mean([df3[!,"RMSE"][1] ]) 

            df_total[!,"MAE_test"][l-1] = mean([df2[!,"MAE"][1] ])#, 
            df_total[!,"MSE_test"][l-1] = mean([df2[!,"MSE"][1] ]) #,
            df_total[!,"NMSE_test"][l-1] = mean([df2[!,"NMSE"][1] ])
            df_total[!,"loss_test"][l-1] = mean([df2[!,"loss"][1] ])
            df_total[!,"RMSE_test"][l-1] = mean([df2[!,"RMSE"][1] ]) #

        end
    	CSV.write(joinpath(saveat, "metrics_train_avg$pretrain.csv"), df_total)
    else
        for l in neurons_test
            df1 = CSV.read(joinpath(saveat, "neurons_$(l)_$layers","metrics_train_$(layers)$pretrain.csv"), DataFrame)
            df2 = CSV.read(joinpath(saveat, "neurons_$(l)_$layers","metrics_test_$(layers)$pretrain.csv"), DataFrame)
            df3 = CSV.read(joinpath(saveat, "neurons_$(l)_$layers","metrics_test_full_$(layers)$pretrain.csv"), DataFrame)

    
            df_total[!,"MAE_training"][l-1] = mean([df1[!,"MAE_train"][1] ])#, df2[!,"MAE_train"][1], df3[!,"MAE_train"][1], df4[!,"MAE_train"][1], df5[!,"MAE_train"][1]])
            df_total[!,"MSE_training"][l-1] = mean([df1[!,"MSE_train"][1] ]) #, df2[!,"MSE_train"][1], df3[!,"MSE_train"][1], df4[!,"MSE_train"][1], df5[!,"MSE_train"][1]])
            df_total[!,"NMSE_training"][l-1] = mean([df1[!,"NMSE_train"][1] ]) 
            df_total[!,"loss_training"][l-1] = mean([df1[!,"loss_train"][1] ]) 
            df_total[!,"RMSE_training"][l-1] = mean([df1[!,"RMSE_train"][1] ]) #, df2[!,"RMSE_train"][1], df3[!,"RMSE_train"][1], df4[!,"RMSE_train"][1], df5[!,"RMSE_train"][1]])

            df_total[!,"MAE_test_binary"][l-1] = mean([df3[!,"MAE"][1] ])#, 
            df_total[!,"MSE_test_binary"][l-1] = mean([df3[!,"MSE"][1] ]) #,
            df_total[!,"NMSE_test_binary"][l-1] = mean([df3[!,"NMSE"][1] ])
            df_total[!,"loss_test_binary"][l-1] = mean([df3[!,"loss"][1] ])
            df_total[!,"RMSE_test_binary"][l-1] = mean([df3[!,"RMSE"][1] ]) 

            df_total[!,"MAE_test"][l-1] = mean([df2[!,"MAE"][1] ])#, 
            df_total[!,"MSE_test"][l-1] = mean([df2[!,"MSE"][1] ]) #,
            df_total[!,"NMSE_test"][l-1] = mean([df2[!,"NMSE"][1] ])
            df_total[!,"loss_test"][l-1] = mean([df2[!,"loss"][1] ])
            df_total[!,"RMSE_test"][l-1] = mean([df2[!,"RMSE"][1] ]) #
        end
        CSV.write(joinpath(saveat, "metrics_train_avg_$(layers)$pretrain.csv"), df_total)
    end
end




function evaluate_metrics_all(saveat, layers=1)
    """
        Take best metrics and put in the train_best/test_best folders
    """	
    if layers == 1
        cp(joinpath(saveat, "neurons_$(df_avg.neurons[idx_min])", "metrics_train.csv"), joinpath(saveat,"train_best", "metrics_train.csv"), force=true)
        cp(joinpath(saveat, "neurons_$(df_avg.neurons[idx_min])", "metrics_test_full.csv"), joinpath(saveat,"test_best", "metrics_test_full.csv"), force=true)
        for i in selection 
            cp(joinpath(saveat, "neurons_$(df_avg.neurons[idx_min])", "plot_training_full_$i.svg"), joinpath(saveat,"test_best", "plot_training_full_$i.svg"), force=true)
            cp(joinpath(saveat, "neurons_$(df_avg.neurons[idx_min])", "plot_training_$i.svg"), joinpath(saveat,"train_best", "plot_training_$i.svg"), force=true)
        end

        for i in ["fig3a", "fig3b", "fig4a", "fig4b"]
            cp(joinpath(saveat, "neurons_$(df_avg.neurons[idx_min])", "test_$i.svg"), joinpath(saveat,"test_best", "test_$i.svg"), force=true)
        end 
        cp(joinpath(saveat, "neurons_$(df_avg.neurons[idx_min])", "metrics_test.csv"), joinpath(saveat,"test_best", "metrics_test.csv"), force=true)

    else
        cp(joinpath(saveat, "neurons_$(df_avg.neurons[idx_min])_$layers", "metrics_train_$layers.csv"), joinpath(saveat,"train_best", "metrics_train_$layers.csv"), force=true)
        cp(joinpath(saveat, "neurons_$(df_avg.neurons[idx_min])_$layers", "metrics_test_full_$layers.csv"), joinpath(saveat,"test_best", "metrics_test_full_$layers.csv"), force=true)
        for i in selection 
            cp(joinpath(saveat, "neurons_$(df_avg.neurons[idx_min])_$layers", "plot_training_full_$(i)_$layers.svg"), joinpath(saveat,"test_best", "plot_training_full_$(i)_$layers.svg"), force=true)
            cp(joinpath(saveat, "neurons_$(df_avg.neurons[idx_min])_$layers", "plot_training_$(i)_$layers.svg"), joinpath(saveat,"train_best", "plot_training_$(i)_$layers.svg"), force=true)
        end

        for i in ["fig3a", "fig3b", "fig4a", "fig4b"]
            cp(joinpath(saveat, "neurons_$(df_avg.neurons[idx_min])_$layers", "test_$(i)_$layers.svg"), joinpath(saveat,"test_best", "test_$(i)_$layers.svg"), force=true)
        end 
        cp(joinpath(saveat, "neurons_$(df_avg.neurons[idx_min])_$layers", "metrics_test_$layers.csv"), joinpath(saveat,"test_best", "metrics_test_$layers.csv"), force=true)
    end

end
