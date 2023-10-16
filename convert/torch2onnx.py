import argparse

import numpy as np
import onnx
from onnx import helper

import torch
from torch import nn
from torchsummary import summary

from mnist_sparse import Net
import spconv.pytorch as spconv

from onnx import NodeProto, TensorProto, GraphProto
from typing import List, Dict

class Torch2OnnxConverter:
  def __init__(self, torch_model, onnx_file):
    self.torch_model : nn.Module = torch_model
    self.onnx_file = onnx_file
    self.var_unique_id = 0
    self.node_unique_id = 0
    self.nodes : List[NodeProto] = []
    self.initializers = []

  
  def var_unique_name(self, name:str):
    uni_name = name + "_" + str(self.var_unique_id)
    self.var_unique_id += 1
    return uni_name

  def node_unique_name(self, name: str):
    uni_name = name + "_" + str(self.node_unique_id)
    self.node_unique_id += 1
    return uni_name

  def _create_initializer_tensor(self, 
      name: str,
      tensor_array: np.ndarray,
      data_type: onnx.TensorProto = onnx.TensorProto.FLOAT) -> onnx.TensorProto:
    initializer_tensor = helper.make_tensor(name=name,
                                            data_type=data_type,
                                            dims=tensor_array.shape,
                                            vals=tensor_array.flatten().tolist())
    return initializer_tensor

  def AddReLU(self, inputs: List[str]):
    out = self.var_unique_name("ReLU")
    self.nodes.append(
        helper.make_node('ReLU',
                        inputs=inputs,
                        outputs=[out],
                        name=self.node_unique_name('ReLU')))
    return [out]


  def AddBatchNorm(self, inputs: List[str], weight_name: List[str], epsilon: float, momentum: float, training_mode: int):
    for w in weight_name:
      weight_init = self._create_initializer_tensor(
          w,
          self.torch_model.get_parameter(w).cpu().detach().numpy())
      self.initializers.append(weight_init)

    mean_name = self.var_unique_name("batch_norm.running_mean")
    batch_norm_1d_running_mean = self._create_initializer_tensor(
        mean_name,
        np.array([0.]).astype(np.float32))
    var_name = self.var_unique_name("batch_norm.running_var")
    batch_norm_1d_running_var = self._create_initializer_tensor(
        var_name,
        np.array([1.]).astype(np.float32))
    self.initializers.append(batch_norm_1d_running_mean)
    self.initializers.append(batch_norm_1d_running_var)

    in_names = []
    in_names.extend(inputs)
    in_names.extend(weight_name)
    in_names.append(mean_name)
    in_names.append(var_name)
    out_name = self.var_unique_name("BatchNormalization")
    node: NodeProto = helper.make_node(
        op_type='BatchNormalization',
        name=self.node_unique_name('BatchNormalization'),
        inputs=in_names,
        outputs=[out_name],
        epsilon=epsilon,
        momentum=momentum,
        training_mode=training_mode)
    self.nodes.append(node)
    return [out_name]


  def AddSubMConv2d(self,
      inputs: List[str],
      weight_name: List[str],
      in_channels: int,
      out_channels: int,
      kernel_size: List[int],
      stride: List[int],
      padding: List[int],
      dilation: List[int],
      output_padding: List[int],
      algo: str,
  ):
    for w in weight_name:
      weight_init = self._create_initializer_tensor(
          w,
          self.torch_model.get_parameter(w).cpu().detach().numpy())
      self.initializers.append(weight_init)

    ins = []
    ins.extend(inputs)
    ins.extend(weight_name)
    out_name = self.var_unique_name("SubMConv2d")
    self.nodes.append(
        helper.make_node(
            op_type='SubMConv2d',
            name=self.node_unique_name("SubMConv2d"),
            inputs=ins,
            outputs=[out_name],
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            output_padding=output_padding,
            algo=algo,
            domain='com.microsoft',
        ))
    return [out_name]


  def AddSparseMaxPool2d(self,
      inputs: List[str],
      kernel_size: List[int],
      stride: List[int],
      padding: List[int],
      dilation: List[int],
      algo: str,
  ):
    out = self.var_unique_name("SparseMaxPool2d")
    self.nodes.append(
        helper.make_node(op_type="SparseMaxPool2d",
                        name=self.node_unique_name("SparseMaxPool2d"),
                        inputs=inputs,
                        outputs=[out],
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        dilation=dilation,
                        algo=algo))
    return [out]

  def AddToDense(self,
      inputs: List[str],
  ):
    out = self.var_unique_name("ToDense")
    self.nodes.append(
        helper.make_node(op_type="ToDense",
                        name=self.node_unique_name("ToDense"),
                        inputs=inputs,
                        outputs=[out]))
    return [out]

  def AddLinear(self, inputs:List[str], weight_name:List[str], alpha, beta, transB):
    for w in weight_name:
      weight_init = self._create_initializer_tensor(
          w,
          self.torch_model.get_parameter(w).cpu().detach().numpy())
      self.initializers.append(weight_init)

    ins = []
    ins.extend(inputs)
    print(ins)
    ins.extend(weight_name)
    out_name = self.var_unique_name("Linear")
    print(ins)
    self.nodes.append(helper.make_node(
      op_type='Gemm',
      name=self.node_unique_name("Linear"),
      inputs=ins,
      outputs=[out_name],
      alpha=alpha,
      beta=beta,
      transB=transB))
    return [out_name]

  def AddLogSoftmax(self, inputs:List[str], axis):
    out_name = self.var_unique_name("LogSoftmax")
    self.nodes.append(helper.make_node(op_type='LogSoftmax',
                                    name=self.node_unique_name('LogSoftmax'),
                                    inputs=[inputs[0]],
                                    outputs=[out_name],
                                    axis=axis))
    return [out_name]


  def Finialize(self, x:List[onnx.ValueInfoProto], out:List[onnx.ValueInfoProto]):
    graph_proto = helper.make_graph(nodes=self.nodes,
                                    name='test_sparse_model',
                                    inputs=x,
                                    outputs=out,
                                    initializer=self.initializers)
    model = helper.make_model(graph_proto)
    # model = onnx.shape_inference.infer_shapes(model)
    # onnx.checker.check_model(model)
    onnx.save(model, self.onnx_file)



def ParseArgs():
  parser = argparse.ArgumentParser(description="torch2onnx example")
  parser.add_argument("torch_model", help="Path to input torch model")
  parser.add_argument("onnx_model", help="Path to output onnx model")
  args = parser.parse_args()
  return args


def load_torch_model():
  model = Net().cuda()
  model_dict = torch.load(args.torch_model)
  model.load_state_dict(model_dict)
  model.eval()
  return model


def convert_to_onnx(torch_model: nn.Module, args):

  converter = Torch2OnnxConverter(torch_model, args.onnx_model)

  batch_norm_out:List[str] = converter.AddBatchNorm(inputs=['x'], weight_name=['net.0.weight', 'net.0.bias'], epsilon=1e-05,
                   momentum=0.1,
                   training_mode=0)
  submconv_out = converter.AddSubMConv2d([batch_norm_out[0]], weight_name=['net.1.weight', 'net.1.bias'], in_channels=1,
                    out_channels=32,
                    kernel_size=[3, 3],
                    stride=[1, 1],
                    padding=[0, 0],
                    dilation=[1, 1],
                    output_padding=[0, 0],
                    algo="MaskImplicitGemm")
  relu_out = converter.AddReLU([submconv_out[0]])
  submconv_out = converter.AddSubMConv2d([relu_out[0]], weight_name=['net.3.weight', 'net.3.bias'],
                    in_channels=32,
                    out_channels=64,
                    kernel_size=[3, 3],
                    stride=[1, 1],
                    padding=[0, 0],
                    dilation=[1, 1],
                    output_padding=[0, 0],
                    algo="MaskImplicitGemm")
  relu_out = converter.AddReLU([submconv_out[0]])
  pool_out = converter.AddSparseMaxPool2d(inputs=[relu_out[0]], kernel_size=[2, 2],
                         stride=[2, 2],
                         padding=[0, 0],
                         dilation=[1, 1],
                         algo="MaskImplicitGemm")
  dense_out = converter.AddToDense([pool_out[0]])
  linear_out = converter.AddLinear([dense_out[0]], ['fc1.weight', 'fc1.bias'], alpha=1, beta=1, transB=1)
  logsoftmax_out = converter.AddLogSoftmax([linear_out[0]], axis=1)

  x = helper.make_sparse_tensor_value_info("x", onnx.TensorProto.FLOAT,
                                           [1 * 28 * 28, 1])
  out = helper.make_sparse_tensor_value_info(logsoftmax_out[0],
                                             onnx.TensorProto.FLOAT, [1, 10])
  converter.Finialize([x], [out])

  for idx, m in enumerate(torch_model.named_modules(remove_duplicate=False)):
    print(idx, m)

  for idx, p in torch_model.named_parameters():
    print(idx, p.size())

  #   if isinstance(m, nn.Linear):
  #     pass
  #   elif isinstance(m, nn.BatchNorm1d):
  #     pass
  #   elif isinstance(m, spconv.SparseConv2d):
  #     pass
  #   elif isinstance(m, spconv.SubMConv2d):
  #     pass
  #   elif isinstance(m, nn.ReLU):
  #     pass
  #   elif isinstance(m, spconv.SparseMaxPool2d):
  #     pass
  #   elif isinstance(m, spconv.ToDense):
  #     pass
  #   elif isinstance(m, nn.Dropout2d):
  #     pass


def main():
  torch_model = load_torch_model()
  summary(torch_model, (28, 28, 1), batch_size=1)
  convert_to_onnx(torch_model, args)
  # x = torch.rand((1, 28, 28, 1)).type(torch.float32).cuda()
  # torch_model(x)


if __name__ == "__main__":
  args = ParseArgs()
  main()
