# reference https://onnxruntime.ai/docs/performance/model-optimizations/float16.html

import argparse
import onnx
from onnxconverter_common import float16
from onnxconverter_common import auto_mixed_precision

def parse_args():
  parser = argparse.ArgumentParser('')
  parser.add_argument('src', type=str, help='src fp32 onnx model')
  parser.add_argument('dst', type=str, help='dst fp16 onnx model')
  return parser.parse_args()


def convert_to_mixed(args):
  model = onnx.load(args.src)
  # model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)

  model_fp16 = float16.convert_float_to_float16(model, min_positive_val=1e-7, max_finite_val=1e4, keep_io_types=True, disable_shape_infer=False, op_block_list=float16.DEFAULT_OP_BLOCK_LIST, node_block_list=None)


  # test_data = {}
  # model_fp16 = auto_convert_mixed_precision(model, test_data, rtol=0.01, atol=0.001, keep_io_types=True)

  onnx.save(model_fp16, args.dst)

if __name__ == "__main__":
  args = parse_args()
  convert_to_mixed(args)

