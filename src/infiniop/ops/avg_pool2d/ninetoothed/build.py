import ninetoothed
from . import avg_pool2d
import infiniop.ninetoothed.build

def build():
    kernel_size_h_values = (7,)
    kernel_size_w_values = (7,)
    stride_h_values = (1, )
    stride_w_values = (1, )
    padding_h_values = (0,)
    padding_w_values = (0,)
    dilation_h_values = (1,)
    dilation_w_values = (1,)
    ceil_mode_values = (False, )
    dtype_values = (ninetoothed.float32,)
    block_size_values = (128,)



    constexpr_param_grid = {
        "kernel_size_h": kernel_size_h_values,
        "kernel_size_w": kernel_size_w_values,
        "stride_h": stride_h_values,
        "stride_w": stride_w_values,
        "padding_h": padding_h_values,
        "padding_w": padding_w_values,
        "dilation_h": dilation_h_values,
        "dilation_w": dilation_w_values,
        "ceil_mode": ceil_mode_values, 
        "dtype": dtype_values,
        "block_size": block_size_values,
    }

    infiniop.ninetoothed.build.build(
        avg_pool2d.premake,
        constexpr_param_grid,
        caller="cuda",
        op_name="avg_pool2d",
        output_dir=infiniop.ninetoothed.build.BUILD_DIRECTORY_PATH,
    )
