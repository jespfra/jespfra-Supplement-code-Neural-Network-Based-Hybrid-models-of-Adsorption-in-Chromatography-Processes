

function evaluate_metrics_validation(saveat, k, l, layers=1, HM = 1)
    """
    A scipt to evaluate the metrics for validation data and plot it. 
        saveat: path to save the results
        k: k-fold split not used anymore 
        l: number of neurons in the hidden layer
        layers: number of hidden layers

    """
    colors = ["blue", "red"]

    # plot pretrain eq data 
    if HM != 0
        p1 = plot()
        for h =1:2
            scatter!(p1, input_pretrain[h,:].*c_scale_data, output_pretrain[:,h], label = "Component $h", color = colors[h])
            plot!(input_pretrain[h,:].*c_scale_data, init_conditions_function(input_pretrain, u0, st, h), label = "Component $h", color = colors[h])
        end
        plot!(p1, xlabel = "Mobile phase concentration / mol/L", ylabel = "Stationary phase concentration / mol/L")
        #display(p1)
        if layers==1
            savefig(p1,joinpath(saveat, "neurons_$l","pretrain_eq_data.svg"))
        else
            savefig(p1,joinpath(saveat, "neurons_$(l)_$layers","pretrain_eq_data.svg"))
        end
    end


    # Get data and Plot train-test for pretrained model i.e. u0 
    inlets, outlets, columns, switches, solverOptions = model()
	model_setup = hybrid_model(columns, columns[1].RHS_q, columns[1].cpp, columns[1].qq, 1, solverOptions.nColumns, solverOptions.idx_units, switches, c_scale_data, q_scale_data)
	solve_model_hybrid(columns=columns, switches=switches, solverOptions=solverOptions, hybrid_model_setup=model_setup, p_NN=u0, outlets=outlets, alg=FBDF(autodiff=false))
	
	
	# Evaluate training metrics
    diff = Float64[]; ndiff = Float64[]; loss = 0.0
    for i = 1:columns[1].nComp 
        append!(diff, abs.(data_train_c[i] .*c_scale_data .- columns[1].solution_outlet[1:test_split,i]))
        append!(ndiff, abs.(data_train_c[i] .- columns[1].solution_outlet[1:test_split,i] ./c_scale_data))
        loss += 1/length(data_train_c[i])^2 * sum(ndiff)^2
	end
	
	MAE = mean(diff)
    MSE = mean(diff.^2)
    NMSE = mean(ndiff.^2)
    RMSE = sqrt(mean(diff.^2))*100
	
	# Save DataFrame to a CSV file
    df = DataFrame(MAE_train = MAE, MSE_train = MSE, NMSE_train = NMSE, loss_train = loss, RMSE_train = RMSE)
    if layers==1
    	CSV.write(joinpath(saveat, "neurons_$l","metrics_train_pretrain.csv"), df)
    else
        CSV.write(joinpath(saveat, "neurons_$(l)_$layers","metrics_train_$(layers)_pretrain.csv"), df)
    end
    
    
    # Evaluate test metrics
    diff = Float64[]; ndiff = Float64[]; loss = 0.0
    
    for i =1:columns[1].nComp  
        append!(diff, abs.(data_test_c[i] .*c_scale_data .- columns[1].solution_outlet[test_split:end,i]))
        append!(ndiff, abs.(data_test_c[i] .- columns[1].solution_outlet[test_split:end,i] ./c_scale_data))
        loss += 1/length(data_test_c[i])^2 * sum(ndiff)^2
    end

    MAE = mean(diff)
    MSE = mean(diff.^2)
    NMSE = mean(ndiff.^2)
    RMSE = sqrt(mean(diff.^2))*100
	
	# Save DataFrame to a CSV file
	df = DataFrame(MAE_test = MAE, MSE_test = MSE, NMSE_test = NMSE, loss_test = loss, RMSE_test = RMSE)
    if layers==1
    	CSV.write(joinpath(saveat, "neurons_$l","metrics_test_pretrain.csv"), df)
    else
        CSV.write(joinpath(saveat, "neurons_$(l)_$layers","metrics_test_$(layers)_pretrain.csv"), df)
    end
	
	
	
	# plot train 
	fig = plot()
    for i =1:columns[1].nComp 
        scatter!(fig, outlets[1].solution_times[1:test_split], data_train_c[i]*c_scale_data, label = "Component $i", seriescolor = colors[i], legend =:topright)
        plot!(fig, outlets[1].solution_times[1:test_split], outlets[1].solution_outlet[1:test_split,i], label = "Component $i", seriescolor = colors[i], legend=:topright, linewidth = 3)
    end
    plot!(legendfontsize = 12, labelfontsize = 12, tickfontsize = 12)
    # display(fig)
    if layers==1
    	savefig(fig,joinpath(saveat, "neurons_$l","train_pretrain.svg"))
    else
	    savefig(fig,joinpath(saveat, "neurons_$(l)_$layers","train_$(layers)_pretrain.svg"))
    end
    
	
	
	# plot test 
	fig = plot()
    for i =1:columns[1].nComp 
        scatter!(fig, outlets[1].solution_times[test_split:end], data_test_c[i].*c_scale_data, label = "Component $i", legend =:right, seriescolor = colors[i])
        plot!(fig, outlets[1].solution_times[test_split:end], outlets[1].solution_outlet[test_split:end,i], label = "Component $i", legend=:right, linewidth = 3, seriescolor = colors[i])
    end
    xlabel!("Time / min")
    ylabel!("Concentration / mol/L")
    plot!(legendfontsize = 12, labelfontsize = 12, tickfontsize = 12)
    # display(fig)
    if layers==1
    	savefig(fig,joinpath(saveat, "neurons_$l","test_pretrain.svg"))
    else
        savefig(fig,joinpath(saveat, "neurons_$(l)_$layers","test_$(layers)_pretrain.svg"))
    end
	
	
	# plot train test 
	fig = plot()
    for i =1:columns[1].nComp 
		scatter!(fig, outlets[1].solution_times[1:end], c_exp_data[:,i], label = "Component $i", legend =:top, seriescolor = colors[i])
        plot!(fig, outlets[1].solution_times[1:end], outlets[1].solution_outlet[1:end,i], label = "Component $i", legend=:top, seriescolor = colors[i], linewidth = 3)
    end
    
    # plot vertical line to indicate training/test split 
    plot!(fig, [outlets[1].solution_times[test_split], outlets[1].solution_times[test_split]], [0, maximum(outlets[1].solution_outlet[:,:])], color = "black", linestyle =:dash, linewidth = 4)
    
    xlabel!("Time / min")
    ylabel!("Concentration / mol/L")
    plot!(legendfontsize = 12, labelfontsize = 12, tickfontsize = 12)
    # display(fig)
    if layers==1
    	savefig(fig,joinpath(saveat, "neurons_$l","train_test_pretrain.svg"))
    else
        savefig(fig,joinpath(saveat, "neurons_$(l)_$layers","train_test_$(layers)_pretrain.svg"))
    end





	# solve model from start to end using trained model 
	inlets, outlets, columns, switches, solverOptions = model()
	model_setup = hybrid_model(columns, columns[1].RHS_q, columns[1].cpp, columns[1].qq, 1, solverOptions.nColumns, solverOptions.idx_units, switches, c_scale_data, q_scale_data)
	solve_model_hybrid(columns=columns, switches=switches, solverOptions=solverOptions, hybrid_model_setup=model_setup, p_NN=NN_model, outlets=outlets, alg=FBDF(autodiff=false))
	
	
	# Evaluate training metrics
    diff = Float64[]; ndiff = Float64[]; loss = 0.0
    for i = 1:columns[1].nComp 
        append!(diff, abs.(data_train_c[i] .*c_scale_data .- columns[1].solution_outlet[1:test_split,i]))
        append!(ndiff, abs.(data_train_c[i] .- columns[1].solution_outlet[1:test_split,i] ./c_scale_data))
        loss += 1/length(data_train_c[i])^2 * sum(ndiff)^2
	end
	
	MAE = mean(diff)
    MSE = mean(diff.^2)
    NMSE = mean(ndiff.^2)
    RMSE = sqrt(mean(diff.^2))*100
	
	# Save DataFrame to a CSV file
    df = DataFrame(MAE_train = MAE, MSE_train = MSE, NMSE_train = NMSE, loss_train = loss, RMSE_train = RMSE)
    if layers==1
    	CSV.write(joinpath(saveat, "neurons_$l","metrics_train.csv"), df)
    else
        CSV.write(joinpath(saveat, "neurons_$(l)_$layers","metrics_train_$layers.csv"), df)
    end
    
    
    # Evaluate test metrics
    diff = Float64[]; ndiff = Float64[]; loss = 0.0
    
    for i =1:columns[1].nComp  
        append!(diff, abs.(data_test_c[i] .*c_scale_data .- columns[1].solution_outlet[test_split:end,i]))
        append!(ndiff, abs.(data_test_c[i] .- columns[1].solution_outlet[test_split:end,i] ./c_scale_data))
        loss += 1/length(data_test_c[i])^2 * sum(ndiff)^2
    end

    MAE = mean(diff)
    MSE = mean(diff.^2)
    NMSE = mean(ndiff.^2)
    RMSE = sqrt(mean(diff.^2))*100
	
	# Save DataFrame to a CSV file
	df = DataFrame(MAE_test = MAE, MSE_test = MSE, NMSE_test = NMSE, loss_test = loss, RMSE_test = RMSE)
    if layers==1
    	CSV.write(joinpath(saveat, "neurons_$l","metrics_test.csv"), df)
    else
        CSV.write(joinpath(saveat, "neurons_$(l)_$layers","metrics_test_$layers.csv"), df)
    end
	
	
	
	# plot train 
	fig = plot()
    for i =1:columns[1].nComp 
        scatter!(fig, outlets[1].solution_times[1:test_split], data_train_c[i]*c_scale_data, label = "Component $i", seriescolor = colors[i], legend =:topright)
        plot!(fig, outlets[1].solution_times[1:test_split], outlets[1].solution_outlet[1:test_split,i], label = "Component $i", seriescolor = colors[i], legend=:topright, linewidth = 3)
    end
    plot!(legendfontsize = 12, labelfontsize = 12, tickfontsize = 12)
    # display(fig)
    if layers==1
    	savefig(fig,joinpath(saveat, "neurons_$l","train.svg"))
    else
	    savefig(fig,joinpath(saveat, "neurons_$(l)_$layers","train_$layers.svg"))
    end
    
	
	
	# plot test 
	fig = plot()
    for i =1:columns[1].nComp 
        scatter!(fig, outlets[1].solution_times[test_split:end], data_test_c[i].*c_scale_data, label = "Component $i", legend =:right, seriescolor = colors[i])
        plot!(fig, outlets[1].solution_times[test_split:end], outlets[1].solution_outlet[test_split:end,i], label = "Component $i", legend=:right, linewidth = 3, seriescolor = colors[i])
    end
    xlabel!("Time / min")
    ylabel!("Concentration / mol/L")
    plot!(legendfontsize = 12, labelfontsize = 12, tickfontsize = 12)
    # display(fig)
    if layers==1
    	savefig(fig,joinpath(saveat, "neurons_$l","test.svg"))
    else
    savefig(fig,joinpath(saveat, "neurons_$(l)_$layers","test_$layers.svg"))
    end
	
	
	# plot train test 
	fig = plot()
    for i =1:columns[1].nComp 
		scatter!(fig, outlets[1].solution_times[1:end], c_exp_data[:,i], label = "Component $i", legend =:top, seriescolor = colors[i])
        plot!(fig, outlets[1].solution_times[1:end], outlets[1].solution_outlet[1:end,i], label = "Component $i", legend=:top, seriescolor = colors[i], linewidth = 3)
    end
    
    # plot vertical line to indicate training/test split 
    plot!(fig, [outlets[1].solution_times[test_split], outlets[1].solution_times[test_split]], [0, maximum(outlets[1].solution_outlet[:,:])], color = "black", linestyle =:dash, linewidth = 4)
    
    xlabel!("Time / min")
    ylabel!("Concentration / mol/L")
    plot!(legendfontsize = 12, labelfontsize = 12, tickfontsize = 12)
    # display(fig)
    if layers==1
    	savefig(fig,joinpath(saveat, "neurons_$l","train_test.svg"))
    else
    savefig(fig,joinpath(saveat, "neurons_$(l)_$layers","train_test_$layers.svg"))
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

function evaluate_metrics_validation_all(saveat, layers=1)
    """
    A scipt to evaluate the metrics for validation data and plot it. 

    """
    df_total = DataFrame(neurons=neurons_test, 
                        MAE_training=zeros(length(neurons_test)), 
                        MSE_training=zeros(length(neurons_test)), 
                        NMSE_training=zeros(length(neurons_test)), 
                        loss_training=zeros(length(neurons_test)),
                        RMSE_training=zeros(length(neurons_test)),
                        
                        MAE_test=zeros(length(neurons_test)), 
                        MSE_test=zeros(length(neurons_test)), 
                        NMSE_test=zeros(length(neurons_test)), 
                        loss_test=zeros(length(neurons_test)),
                        RMSE_test=zeros(length(neurons_test)),
                        
                        MAE_training_pretrain=zeros(length(neurons_test)), 
                        MSE_training_pretrain=zeros(length(neurons_test)), 
                        NMSE_training_pretrain=zeros(length(neurons_test)), 
                        loss_training_pretrain=zeros(length(neurons_test)),
                        RMSE_training_pretrain=zeros(length(neurons_test)),
                        
                        MAE_test_pretrain=zeros(length(neurons_test)), 
                        MSE_test_pretrain=zeros(length(neurons_test)), 
                        NMSE_test_pretrain=zeros(length(neurons_test)), 
                        loss_test_pretrain=zeros(length(neurons_test)),
                        RMSE_test_pretrain=zeros(length(neurons_test)))
    

                        
    if layers==1
        for l in neurons_test
            df1 = CSV.read(joinpath(saveat, "neurons_$l","metrics_train.csv"), DataFrame)
            df2 = CSV.read(joinpath(saveat, "neurons_$l","metrics_test.csv"), DataFrame)
            
    
            df_total[!,"MAE_training"][l-1] = mean([df1[!,"MAE_train"][1] ])#, df2[!,"MAE_train"][1], df3[!,"MAE_train"][1], df4[!,"MAE_train"][1], df5[!,"MAE_train"][1]])
            df_total[!,"MSE_training"][l-1] = mean([df1[!,"MSE_train"][1] ]) #, df2[!,"MSE_train"][1], df3[!,"MSE_train"][1], df4[!,"MSE_train"][1], df5[!,"MSE_train"][1]])
            df_total[!,"NMSE_training"][l-1] = mean([df1[!,"NMSE_train"][1] ]) 
            df_total[!,"loss_training"][l-1] = mean([df1[!,"loss_train"][1] ]) 
            df_total[!,"RMSE_training"][l-1] = mean([df1[!,"RMSE_train"][1] ])

            df_total[!,"MAE_test"][l-1] = mean([df2[!,"MAE_test"][1] ])#, 
            df_total[!,"MSE_test"][l-1] = mean([df2[!,"MSE_test"][1] ]) #,
            df_total[!,"NMSE_test"][l-1] = mean([df2[!,"NMSE_test"][1] ])
            df_total[!,"loss_test"][l-1] = mean([df2[!,"loss_test"][1] ])
            df_total[!,"RMSE_test"][l-1] = mean([df2[!,"RMSE_test"][1] ])


            # pretrained model only 
            df3 = CSV.read(joinpath(saveat, "neurons_$l","metrics_train_pretrain.csv"), DataFrame)
            df4 = CSV.read(joinpath(saveat, "neurons_$l","metrics_test_pretrain.csv"), DataFrame)

            df_total[!,"MAE_training_pretrain"][l-1] = mean([df3[!,"MAE_train"][1] ])#, df2[!,"MAE_train"][1], df3[!,"MAE_train"][1], df4[!,"MAE_train"][1], df5[!,"MAE_train"][1]])
            df_total[!,"MSE_training_pretrain"][l-1] = mean([df3[!,"MSE_train"][1] ]) #, df2[!,"MSE_train"][1], df3[!,"MSE_train"][1], df4[!,"MSE_train"][1], df5[!,"MSE_train"][1]])
            df_total[!,"NMSE_training_pretrain"][l-1] = mean([df3[!,"NMSE_train"][1] ]) 
            df_total[!,"loss_training_pretrain"][l-1] = mean([df3[!,"loss_train"][1] ]) 
            df_total[!,"RMSE_training_pretrain"][l-1] = mean([df3[!,"RMSE_train"][1] ])

            df_total[!,"MAE_test_pretrain"][l-1] = mean([df4[!,"MAE_test"][1] ])#, 
            df_total[!,"MSE_test_pretrain"][l-1] = mean([df4[!,"MSE_test"][1] ]) #,
            df_total[!,"NMSE_test_pretrain"][l-1] = mean([df4[!,"NMSE_test"][1] ])
            df_total[!,"loss_test_pretrain"][l-1] = mean([df4[!,"loss_test"][1] ])
            df_total[!,"RMSE_test_pretrain"][l-1] = mean([df4[!,"RMSE_test"][1] ])
        end
    	CSV.write(joinpath(saveat, "metrics_train_avg.csv"), df_total)
    else
        for l in neurons_test
            df1 = CSV.read(joinpath(saveat, "neurons_$(l)_$layers","metrics_train_$layers.csv"), DataFrame)
            df2 = CSV.read(joinpath(saveat, "neurons_$(l)_$layers","metrics_test_$layers.csv"), DataFrame)

    
            df_total[!,"MAE_training"][l-1] = mean([df1[!,"MAE_train"][1] ])#, df2[!,"MAE_train"][1], df3[!,"MAE_train"][1], df4[!,"MAE_train"][1], df5[!,"MAE_train"][1]])
            df_total[!,"MSE_training"][l-1] = mean([df1[!,"MSE_train"][1] ]) #, df2[!,"MSE_train"][1], df3[!,"MSE_train"][1], df4[!,"MSE_train"][1], df5[!,"MSE_train"][1]])
            df_total[!,"NMSE_training"][l-1] = mean([df1[!,"NMSE_train"][1] ]) 
            df_total[!,"loss_training"][l-1] = mean([df1[!,"loss_train"][1] ]) 
            df_total[!,"RMSE_training"][l-1] = mean([df1[!,"RMSE_train"][1] ])

            df_total[!,"MAE_test"][l-1] = mean([df2[!,"MAE_test"][1] ])#, 
            df_total[!,"MSE_test"][l-1] = mean([df2[!,"MSE_test"][1] ]) #,
            df_total[!,"NMSE_test"][l-1] = mean([df2[!,"NMSE_test"][1] ])
            df_total[!,"loss_test"][l-1] = mean([df2[!,"loss_test"][1] ])
            df_total[!,"RMSE_test"][l-1] = mean([df2[!,"RMSE_test"][1] ])

            # pretrained model only 
            df3 = CSV.read(joinpath(saveat, "neurons_$(l)_$layers","metrics_train_$(layers)_pretrain.csv"), DataFrame)
            df4 = CSV.read(joinpath(saveat, "neurons_$(l)_$layers","metrics_test_$(layers)_pretrain.csv"), DataFrame)
 
            df_total[!,"MAE_training_pretrain"][l-1] = mean([df3[!,"MAE_train"][1] ])#, df2[!,"MAE_train"][1], df3[!,"MAE_train"][1], df4[!,"MAE_train"][1], df5[!,"MAE_train"][1]])
            df_total[!,"MSE_training_pretrain"][l-1] = mean([df3[!,"MSE_train"][1] ]) #, df2[!,"MSE_train"][1], df3[!,"MSE_train"][1], df4[!,"MSE_train"][1], df5[!,"MSE_train"][1]])
            df_total[!,"NMSE_training_pretrain"][l-1] = mean([df3[!,"NMSE_train"][1] ]) 
            df_total[!,"loss_training_pretrain"][l-1] = mean([df3[!,"loss_train"][1] ]) 
            df_total[!,"RMSE_training_pretrain"][l-1] = mean([df3[!,"RMSE_train"][1] ])

            df_total[!,"MAE_test_pretrain"][l-1] = mean([df4[!,"MAE_test"][1] ])#, 
            df_total[!,"MSE_test_pretrain"][l-1] = mean([df4[!,"MSE_test"][1] ]) #,
            df_total[!,"NMSE_test_pretrain"][l-1] = mean([df4[!,"NMSE_test"][1] ])
            df_total[!,"loss_test_pretrain"][l-1] = mean([df4[!,"loss_test"][1] ])
            df_total[!,"RMSE_test_pretrain"][l-1] = mean([df4[!,"RMSE_test"][1] ])
        end
        CSV.write(joinpath(saveat, "metrics_train_avg_$layers.csv"), df_total)
    end


end


function evaluate_metrics_all(saveat, layers=1)
    """
    A scipt to evaluate the metrics for all training and test data.

    """
	if layers == 1
        # trained model
        cp(joinpath(saveat, "neurons_$(df_avg.neurons[idx_min])", "metrics_train.csv"), joinpath(saveat,"train_best", "metrics_train.csv"), force=true)
        cp(joinpath(saveat, "neurons_$(df_avg.neurons[idx_min])", "metrics_test.csv"), joinpath(saveat,"test_best", "metrics_test.csv"), force=true)
        cp(joinpath(saveat, "neurons_$(df_avg.neurons[idx_min])", "train_test.svg"), joinpath(saveat,"test_best", "train_test.svg"), force=true)
        cp(joinpath(saveat, "neurons_$(df_avg.neurons[idx_min])", "train.svg"), joinpath(saveat,"train_best", "train.svg"), force=true)
        cp(joinpath(saveat, "neurons_$(df_avg.neurons[idx_min])", "test.svg"), joinpath(saveat,"test_best", "test.svg"), force=true)

        # pretrained model
        cp(joinpath(saveat, "neurons_$(df_avg.neurons[idx_min])", "metrics_train_pretrain.csv"), joinpath(saveat,"train_best", "metrics_train_pretrain.csv"), force=true)
        cp(joinpath(saveat, "neurons_$(df_avg.neurons[idx_min])", "metrics_test_pretrain.csv"), joinpath(saveat,"test_best", "metrics_test_pretrain.csv"), force=true)
        cp(joinpath(saveat, "neurons_$(df_avg.neurons[idx_min])", "train_test_pretrain.svg"), joinpath(saveat,"test_best", "train_test_pretrain.svg"), force=true)
        cp(joinpath(saveat, "neurons_$(df_avg.neurons[idx_min])", "train_pretrain.svg"), joinpath(saveat,"train_best", "train_pretrain.svg"), force=true)
        cp(joinpath(saveat, "neurons_$(df_avg.neurons[idx_min])", "test_pretrain.svg"), joinpath(saveat,"test_best", "test_pretrain.svg"), force=true)

    else
        # trained model
        cp(joinpath(saveat, "neurons_$(df_avg.neurons[idx_min])_$layers", "metrics_train_$layers.csv"), joinpath(saveat,"train_best", "metrics_train_$layers.csv"), force=true)
        cp(joinpath(saveat, "neurons_$(df_avg.neurons[idx_min])_$layers", "metrics_test_$layers.csv"), joinpath(saveat,"test_best", "metrics_test_$layers.csv"), force=true)
        cp(joinpath(saveat, "neurons_$(df_avg.neurons[idx_min])_$layers", "train_test_$layers.svg"), joinpath(saveat,"test_best", "train_test_$layers.svg"), force=true)
        cp(joinpath(saveat, "neurons_$(df_avg.neurons[idx_min])_$layers", "train_$layers.svg"), joinpath(saveat,"train_best", "train_$layers.svg"), force=true)
        cp(joinpath(saveat, "neurons_$(df_avg.neurons[idx_min])_$layers", "test_$layers.svg"), joinpath(saveat,"test_best", "test_$layers.svg"), force=true)

        # pretrained model
        cp(joinpath(saveat, "neurons_$(df_avg.neurons[idx_min])_$layers", "metrics_train_$(layers)_pretrain.csv"), joinpath(saveat,"train_best", "metrics_train_$(layers)_pretrain.csv"), force=true)
        cp(joinpath(saveat, "neurons_$(df_avg.neurons[idx_min])_$layers", "metrics_test_$(layers)_pretrain.csv"), joinpath(saveat,"test_best", "metrics_test_$(layers)_pretrain.csv"), force=true)
        cp(joinpath(saveat, "neurons_$(df_avg.neurons[idx_min])_$layers", "train_test_$(layers)_pretrain.svg"), joinpath(saveat,"test_best", "train_test_$(layers)_pretrain.svg"), force=true)
        cp(joinpath(saveat, "neurons_$(df_avg.neurons[idx_min])_$layers", "train_$(layers)_pretrain.svg"), joinpath(saveat,"train_best", "train_$(layers)_pretrain.svg"), force=true)
        cp(joinpath(saveat, "neurons_$(df_avg.neurons[idx_min])_$layers", "test_$(layers)_pretrain.svg"), joinpath(saveat,"test_best", "test_$(layers)_pretrain.svg"), force=true)
        
    end
	

end
