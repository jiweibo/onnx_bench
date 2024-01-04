import pathlib
from typing import List
import numpy as np
import onnxruntime as ort


class Session:

    def __init__(
        self,
        onnx_model: str,
        provider: str,
        precision: str = "fp32",
        cache_dir: str = None,
        trt_min_subgraph_size: int = 1,
        filter_ops: str = None,
        device_id: int = 0,
    ):
        if not pathlib.Path(onnx_model).exists():
            raise ValueError(onnx_model + " not exists")
        self.device_type = "cpu"
        self.provider = provider
        self.precision = precision
        self.cache_dir = cache_dir
        self.trt_min_subgraph_size = trt_min_subgraph_size
        self.filter_ops = filter_ops
        self.device_id = device_id

        self.onnx_model = onnx_model
        self.create_session()
        self.init_sess()

    def create_session(self):
        providers = ["CPUExecutionProvider"]
        if self.provider == "cuda":
            providers.insert(0, ("CUDAExecutionProvider", {
                "device_id": self.device_id
            }))
        if self.provider == "trt":
            providers.insert(0, ("CUDAExecutionProvider", {
                "device_id": self.device_id
            }))
            providers.insert(
                0,
                (
                    "TensorrtExecutionProvider",
                    {
                        "device_id": self.device_id,
                        "trt_max_workspace_size": 1 << 31,
                        "trt_fp16_enable": self.precision == "fp16",
                        "trt_engine_cache_enable": self.cache_dir != None,
                        "trt_engine_cache_path": self.cache_dir,
                        # 'trt_max_partition_iterations':1000,
                        "trt_min_subgraph_size": self.trt_min_subgraph_size,
                        'trt_filter_ops': self.filter_ops,
                        # 'trt_force_fp32_ops': 'MatMul_154 MatMul_171 Concat_194 ReduceMean_360 Sqrt_363',
                    },
                ),
            )

        self.sess = ort.InferenceSession(self.onnx_model, providers=providers)

    def init_sess(self):
        print("-------- input_info --------")
        self.input_names = []
        self.input_types = []
        self.input_shapes = []
        for x in self.sess.get_inputs():
            print(x)
            self.input_names.append(x.name)
            self.input_types.append(x.type)
            self.input_shapes.append(x.shape)

        print("-------- output_info --------")
        self.output_names = []
        for x in self.sess.get_outputs():
            print(x)
            self.output_names.append(x.name)

        self.io_bind = self.sess.io_binding()

    def run(self, ins: List[np.ndarray]):
        assert len(ins) == len(self.input_names)
        self.io_bind.clear_binding_inputs()
        self.io_bind.clear_binding_outputs()

        for name, data in zip(self.input_names, ins):
            x_ortvalue = ort.OrtValue.ortvalue_from_numpy(
                data, device_type=self.device_type, device_id=self.device_id)
            self.io_bind.bind_input(
                name=name,
                device_type=x_ortvalue.device_name(),
                device_id=0,
                element_type=data.dtype,
                shape=data.shape,
                buffer_ptr=x_ortvalue.data_ptr(),
            )

        for name in self.output_names:
            self.io_bind.bind_output(name=name,
                                     device_type=self.device_type,
                                     device_id=self.device_id)

        self.sess.run_with_iobinding(self.io_bind)

        outs = self.io_bind.get_outputs()
        return [out.numpy() for out in outs]
