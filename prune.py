import onnx
import argparse
from shutil import copyfile

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("src", type=str, help="src onnx")
    parser.add_argument("dst", type=str, help="dst onnx")
    parser.add_argument("--output_nodes", type=str, default=None, help="output node: a,b,c, default None")
    parser.add_argument("--input_nodes", type=str, default=None, help="input node: a,b,c, default None")
    parser.add_argument("--no_origin_in_nodes", action="store_false", help="add origin input nodes or not")
    parser.add_argument("--origin_out_nodes", action="store_true", help="add origin output nodes or not, only useful when --prune")
    parser.add_argument("--prune", action="store_true", help="prune model or just mark some tensors output node.")
    return parser.parse_args()

def get_origin_input_output_names(model):
    num = len(model.graph.input)
    input_names = []
    for i in range(num):
        input_names.append(model.graph.input[i].name)

    num = len(model.graph.output)
    output_names = []
    for i in range(num):
        output_names.append(model.graph.output[i].name)

    return input_names, output_names


if __name__ == "__main__":
    args = parse()

    model = onnx.load(args.src)
    ori_in_names, ori_out_names = get_origin_input_output_names(model)
    real_input_names = []
    real_output_names = []

    if args.input_nodes is not None:
        real_input_names = args.input_nodes.split(",")
    if args.no_origin_in_nodes:
        real_input_names.extend(ori_in_names)
    print("input_nodes: ", real_input_names)

    if args.output_nodes is not None:
        real_output_names = args.output_nodes.split(",")


    if args.prune:
        if args.origin_out_nodes:
            real_output_names.extend(ori_out_names)

        print("output_nodes: ", real_output_names)
        onnx.utils.extract_model(args.src, args.dst, real_input_names, real_output_names)
    else:
        value_info_protos = []
        print("output_nodes: ", real_output_names+ori_out_names)
        for idx, node in enumerate(model.graph.value_info):
            if node.name in real_output_names and node.name not in ori_out_names:
                value_info_protos.append(node)
        model.graph.output.extend(value_info_protos)
        onnx.checker.check_model(model)
        onnx.save(model, args.dst)