import ninetoothed
from . import conv2d
import infiniop.ninetoothed.build



def build():
    # input_precision_values=(1,)
    # stride_h_values=(1,2)
    # stride_w_values=(1,2)
    # padding_h_values=(0,1,3)
    # padding_w_values=(0,1,3)
    # dilation_h_values=(1,)
    # dilation_w_values=(1,)

    # dtype_values=(ninetoothed.float32, )
    # block_size_m_values=(32,)
    # block_size_n_values=(32,)
    # block_size_k_values=(32,)
    pass

    # input_precision_values=(1,)
    # stride_h_values=(1,)
    # stride_w_values=(1,)
    # padding_h_values=(0,)
    # padding_w_values=(0,)
    # dilation_h_values=(1,)
    # dilation_w_values=(1,)

    # dtype_values=(ninetoothed.float32, )
    # block_size_m_values=(32,)
    # block_size_n_values=(32,)
    # block_size_k_values=(32,)

    # constexpr_param_grid = {
    #     "input_precision": input_precision_values,
    #     "stride_h": stride_h_values,
    #     "stride_w": stride_w_values,
    #     "padding_h": padding_h_values,
    #     "padding_w": padding_w_values,
    #     "dilation_h": dilation_h_values,
    #     "dilation_w": dilation_w_values,
    #     "dtype": dtype_values,
    #     "block_size_m": block_size_m_values,
    #     "block_size_n": block_size_n_values,
    #     "block_size_k": block_size_k_values,
    # }

    # infiniop.ninetoothed.build.build(
    #     conv2d.premake,
    #     constexpr_param_grid,
    #     caller="cuda",
    #     op_name="conv2d",
    #     output_dir=infiniop.ninetoothed.build.BUILD_DIRECTORY_PATH,
    # )