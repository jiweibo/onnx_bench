import numpy as np
import json
import argparse
import os
from pathlib import Path


def parse_args():
  parser = argparse.ArgumentParser('')
  parser.add_argument('src_dir', type=str, help='src json dir')
  parser.add_argument('dst_dir', type=str, help='dst npz dir')
  return parser.parse_args()


def json_to_np(json_file):
    """
    将包含张量信息的JSON字符串转换为NP
    
    参数:
    json_file (str): 包含张量信息的JSON
    """
    # 转换数据类型
    dtype_map = {
        "kFloat": np.float32,
        "int32": np.int32,
        "int64": np.int64,
        "int8": np.int8,
        "kBool": bool,
    }
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        tensor_info = data["data"][0]
        name = tensor_info["tensor_name"]
        
        dtype = dtype_map.get(tensor_info["tensor_type"], None)
        
        # 创建numpy数组
        np_array = np.array(tensor_info["tensor_data"], dtype=dtype)
        
        # 应用维度
        if tensor_info["dims"]:
            try:
                np_array = np_array.reshape(tensor_info["dims"])
            except ValueError as e:
                print(f"维度不匹配: {name} {e}")
                exit(2)

        return name, np_array
            
    except Exception as e:
        print(f"转换失败: {str(e)}")
        exit(2)


def main(args):
  srcs = os.listdir(args.src_dir)
  srcs = sorted(srcs)

  last_npz_name = None
  tensor_dic = {}

  for i in range(len(srcs)):
    src = srcs[i]
    if 'input' not in src:
      continue
    last_dash_index = src.rfind('.')
    npz_name = src[:last_dash_index]
    if i == 0:
      last_npz_name = npz_name
    if npz_name != last_npz_name:
      np.savez(os.path.join(args.dst_dir, last_npz_name+'.npz'), **tensor_dic)
      last_npz_name = npz_name
      tensor_dic = {}

    name, arr = json_to_np(os.path.join(args.src_dir, src))
    tensor_dic[name] = arr


if __name__ == '__main__':
  args = parse_args()
  main(args)