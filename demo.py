
import argparse

import numpy as np
import onnxruntime as ort

np.random.seed(1998)

def parse():
    parser = argparse.ArgumentParser('')
    parser.add_argument('onnx', type=str, help='')
    parser.add_argument('--batch', type=int, default=1, help='')
    return parser.parse_args()

def generate_ort_data(shape, dtype):
    if dtype == 'tensor(float)':
        return np.random.rand(*shape).astype(np.float32)
    elif dtype == 'tensor(int32)':
        return np.random.randint(-128, 127, shape).astype(np.int32)
    elif dtype == 'tensor(bool)':
        return np.random.randint(0, 1, shape).astype(np.bool)
    else:
        raise NotImplementedError('not support for %s' % dtype)


def main(onnx_model, args):
    session = ort.InferenceSession(onnx_model, providers=[
        ("TensorrtExecutionProvider", {
            'device_id': 0,
            'trt_max_workspace_size': 1073741824,
            'trt_fp16_enable': True,
            'trt_max_partition_iterations':1000,
            'trt_min_subgraph_size':3,
            'trt_filter_ops':'ReduceMax_188',
            'trt_force_fp32_ops':'MatMul_154 MatMul_171 Concat_194 ReduceMean_360 Sqrt_363',
        }),
        ("CUDAExecutionProvider", {
            'device_id': 0,
        }),
        'CPUExecutionProvider'
    ])


    # io_binding = session.io_binding()
    ins = {}
    for x in session.get_inputs():
        x_shape = x.shape
        # Just for input shape, need a better way.
        if isinstance(x.shape[0], str) or x.shape[0] == -1:
            x_shape[0] = args.batch

        np_data = generate_ort_data(x_shape, x.type)
        ins[x.name] = np_data
        # ortval = ort.OrtValue.ortvalue_from_numpy(np_data, 'cpu', 0)
        # io_binding.bind_input(name=x.name, device_type=ortval.device_name(), device_id=0, element_type=np_data.dtype, shape=x_shape, buffer_ptr=ortval.data_ptr())
    
    # for y in session.get_outpus():
    out_names = [o.name for o in session.get_outputs()]
    out = session.run(None, ins)

    print(out)


    # session.run_with_iobinding(io_binding)





if __name__ == "__main__":
    args = parse()
    main(args.onnx, args)


