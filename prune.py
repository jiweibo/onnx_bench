import onnx
import argparse
from shutil import copyfile

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("src", type=str, help="src onnx")
    parser.add_argument("dst", type=str, help="dst onnx")
    parser.add_argument("--output_nodes", type=str, default=None, help="output node: a,b,c")
    parser.add_argument("--input_nodes", type=str, default=None, help="input node: a,b,c")
    return parser.parse_args()

def parse_onnx_inputs(filename):
    model = onnx.load(filename)
    num = len(model.graph.input)
    input_names = []
    for i in range(num):
        input_names.append(model.graph.input[i].name)
    return ",".join(input_names)

def parse_onnx_outputs(filename):
    model = onnx.load(filename)
    num = len(model.graph.output)
    output_names = []
    for i in range(num):
        output_names.append(model.graph.output[i].name)


    # for binary.
    # all_nodes = model.graph.node
    # node_len = len(all_nodes)
    # down_binary_names = all_nodes[int(node_len/2)].output

    return ",".join(output_names)


if __name__ == "__main__":
    args = parse()

    base_input_nodes = parse_onnx_inputs(args.src).split(",")
    base_output_nodes = parse_onnx_outputs(args.src).split(",")
    input_nodes = []
    output_nodes = []

    skip_prune = True
    if args.input_nodes is not None:
        input_nodes = args.input_nodes.split(",")
        if input_nodes != base_input_nodes:
            skip_prune = False
    else:
        input_nodes = base_input_nodes

    if args.output_nodes is not None:
        output_nodes = args.output_nodes.split(",")
        if output_nodes != base_output_nodes:
            skip_prune = False
        
    print("input_nodes: ", input_nodes)
    print("output_nodes: ", output_nodes)
    
    if skip_prune:
        copyfile(args.src, args.dst)
    else:
        onnx.utils.extract_model(args.src, args.dst, input_nodes, output_nodes)