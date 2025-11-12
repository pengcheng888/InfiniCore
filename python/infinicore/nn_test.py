import infinicore


def func6():
    import infinicore

    import torch
    
    input = torch.ones(5,5) 
    weight = torch.ones(5,5) 
    bias = torch.ones(5,) 


    input_infini = infinicore.experimental.convert_torch_to_infini_tensor(input)
    weight_infini = infinicore.experimental.convert_torch_to_infini_tensor(weight)

    if bias   is not None:
        bias_infini = infinicore.experimental.convert_torch_to_infini_tensor(bias)
    else:
        bias_infini = None

    
    y_infini  = infinicore.nn.functional.linear(input_infini, weight_infini, bias_infini)

    print(y_infini)

    y_torch = torch.nn.functional.linear(input, weight, bias)

    print(y_torch)


def func7():
    import infinicore

    import torch
    
    vocab_size = 10
    embedding_dim = 3 

    input = torch.ones(1,5,dtype=torch.int32) 
    weight = torch.ones(vocab_size,embedding_dim) 

    output = torch.ones(1,5,embedding_dim) 

    input_infini = infinicore.experimental.convert_torch_to_infini_tensor(input)
    weight_infini = infinicore.experimental.convert_torch_to_infini_tensor(weight)
    output_infini = infinicore.experimental.convert_torch_to_infini_tensor(output)

    
    y_infini  = infinicore.nn.functional.embedding(input_infini, 
                                                   weight_infini,
                                                   out = output_infini)

    print(y_infini)


def func7():
    from infinicore.lib import _infinicore
    print(    _infinicore.Algo.GPT_NEOX)

    import torch
    bs, ntok, head_dim = 5,32,64
    
    x =   torch.ones(( bs,ntok, head_dim))
    pos_ids =  torch.tensor([1,2,3,4,5],dtype=torch.int32) 
    sin_table= torch.ones(( 2048, head_dim//2)) 
    cos_table=  torch.ones(( 2048, head_dim//2))

    algo =  _infinicore.Algo.GPT_NEOX

    x_infini = infinicore.experimental.convert_torch_to_infini_tensor( x)
    pos_ids_infini = infinicore.experimental.convert_torch_to_infini_tensor(pos_ids)
    sin_table_infini = infinicore.experimental.convert_torch_to_infini_tensor(sin_table)
    cos_table_infini = infinicore.experimental.convert_torch_to_infini_tensor(cos_table)
    
    y = infinicore.nn.functional.rope(x_infini,pos_ids_infini,sin_table_infini,cos_table_infini,algo)
    print(y)

if __name__ == '__main__':
    #func7()
    pass
