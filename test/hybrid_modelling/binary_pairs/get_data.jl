

using CSV, DataFrames
function get_data()
	"""
		A function to get the correct data for training & test
	"""
	# load data
	# Pairswise data  
	AcidPro_ProPro_data = CSV.read(joinpath(@__DIR__, "data", "AcidPro_ProPro", "binary.csv"), DataFrame; types=Dict(1 => Float64, 3 => Float64),validate=false)
	H2O_AcidPro_data = CSV.read(joinpath(@__DIR__, "data", "H2O_AcidPro", "binary.csv"), DataFrame; types=Dict(1 => Float64, 4 => Float64),validate=false)
	Poh_ProPro_data = CSV.read(joinpath(@__DIR__, "data", "Poh_ProPro", "binary.csv"), DataFrame; types=Dict(2 => Float64, 3 => Float64),validate=false)
	
	# eq
	c_scale_data = [13.2, 55, 13.36] # max concentration
	q_scale_data = [9.24, 27, 8.1] # max q
	AcidPro_ProPro_data_eq = CSV.read(joinpath(@__DIR__, "data", "AcidPro_ProPro", "eq_data.csv"), DataFrame) 
	H2O_AcidPro_data_eq = CSV.read(joinpath(@__DIR__, "data", "H2O_AcidPro", "eq_data.csv"), DataFrame; types=Dict(1 => Float64, 4 => Float64),validate=false) 
	Poh_ProPro_data_eq = CSV.read(joinpath(@__DIR__, "data", "Poh_ProPro", "eq_data.csv"), DataFrame; types=Dict(2 => Float64, 3 => Float64),validate=false)

	# Pretrain data 
	# The input is always scaled to 55 as it is the highest concentration in the data
	input_pretrain = vcat(Array(AcidPro_ProPro_data_eq[:, 1:4])./c_scale_data[2], Array(H2O_AcidPro_data_eq[:, 1:4])./c_scale_data[2], Array(Poh_ProPro_data_eq[:, 1:4])./c_scale_data[2])'
	output_pretrain = vcat(Array(AcidPro_ProPro_data_eq[:, 5:8]), Array(H2O_AcidPro_data_eq[:, 5:8]), Array(Poh_ProPro_data_eq[:, 5:8])) 

	# Load training data 
	AcidPro_ProPro_data_eq[:,5:end] ./= q_scale_data[1]	
	H2O_AcidPro_data_eq[:,5:end] ./= q_scale_data[2]
	Poh_ProPro_data_eq[:,5:end] ./= q_scale_data[3]

	AcidPro_ProPro_data_train_split = 98 
	AcidPro_ProPro_data_train = AcidPro_ProPro_data[1:AcidPro_ProPro_data_train_split, :]
	AcidPro_ProPro_data_train[:, 1:4] ./= c_scale_data[1]

	H2O_AcidPro_data_train_split = 162
	H2O_AcidPro_data_train = H2O_AcidPro_data[1:H2O_AcidPro_data_train_split, :] 
	H2O_AcidPro_data_train[:, 1:4] ./= c_scale_data[2]

	Poh_ProPro_data_train_split = 204
	Poh_ProPro_data_train = Poh_ProPro_data[1:Poh_ProPro_data_train_split, :]
	Poh_ProPro_data_train[:, 1:4] ./= c_scale_data[3]

	# load test data 
	AcidPro_ProPro_data_test = AcidPro_ProPro_data[AcidPro_ProPro_data_train_split:end, :]
	H2O_AcidPro_data_test = H2O_AcidPro_data[H2O_AcidPro_data_train_split:end, :] 
	Poh_ProPro_data_test = Poh_ProPro_data[Poh_ProPro_data_train_split:end, :] 

	# load reactive data 
    c_exp_data3a = CSV.read(joinpath(@__DIR__, "data", "reactive", "fig3a.csv"), DataFrame;)
	c_exp_data3b = CSV.read(joinpath(@__DIR__, "data", "reactive", "fig3b.csv"), DataFrame;)
	c_exp_data4a = CSV.read(joinpath(@__DIR__, "data", "reactive", "fig4a.csv"), DataFrame;)
	c_exp_data4b = CSV.read(joinpath(@__DIR__, "data", "reactive", "fig4b.csv"), DataFrame;)

	data_trains = [AcidPro_ProPro_data_train, H2O_AcidPro_data_train, Poh_ProPro_data_train]
	data_full = [AcidPro_ProPro_data, H2O_AcidPro_data, Poh_ProPro_data]
	data_train_eq = [AcidPro_ProPro_data_eq[:,5:end], H2O_AcidPro_data_eq[:,5:end], Poh_ProPro_data_eq[:,5:end]] # scaled q data so it does not weight higher in loss function 

	

    
    return c_scale_data, q_scale_data, input_pretrain, output_pretrain, data_trains, data_full, data_train_eq, AcidPro_ProPro_data, H2O_AcidPro_data, Poh_ProPro_data, AcidPro_ProPro_data_train, H2O_AcidPro_data_train, Poh_ProPro_data_train, AcidPro_ProPro_data_test, H2O_AcidPro_data_test, Poh_ProPro_data_test, AcidPro_ProPro_data_eq, H2O_AcidPro_data_eq, Poh_ProPro_data_eq, c_exp_data3a, c_exp_data3b, c_exp_data4a, c_exp_data4b
end
