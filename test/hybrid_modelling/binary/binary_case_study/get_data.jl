

using CSV, DataFrames
function get_data(; test_split, c_scale_data="max", q_scale_data="max")
	"""
		A function to get the correct data for training 
	"""
    
    # load data 
    c_exp_data = CSV.read(joinpath(@__DIR__, "data", "binary.csv"), DataFrame;)
    qc_exp_data = CSV.read(joinpath(@__DIR__, "data", "conditions.csv"), DataFrame;)

    if c_scale_data == "max"
        c_scale_data = maximum(Array(c_exp_data[:,1:2]))
    end
    if q_scale_data == "max"
        q_scale_data = maximum(Array(qc_exp_data[:,6:7]))
    end
    
    #Setting up training data
    data_train_c = (c_exp_data[1:test_split, 1]/c_scale_data, c_exp_data[1:test_split, 2]/c_scale_data)
    data_train_q = Array(qc_exp_data[1:end-1,6:7]) / q_scale_data

    data_test_c = (c_exp_data[test_split:end, 1]/c_scale_data, c_exp_data[test_split:end, 2]/c_scale_data)
    data_test_q = Array(qc_exp_data[end,6:7]) / q_scale_data

    input_pretrain = Array(qc_exp_data[1:end-1, 4:5])' ./ c_scale_data
    output_pretrain = Array(qc_exp_data[1:end-1, 6:7])
    
    return c_exp_data, qc_exp_data, input_pretrain, output_pretrain, data_train_c, data_train_q, data_test_c, data_test_q, c_scale_data, q_scale_data
end