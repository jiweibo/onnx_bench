set -x
set -e

DIR="$(cd "$(dirname "$0")" ; pwd -P)"
cd "$DIR"

model=$1
precision=$2
node_names=$3
base_provider=${4:-cpu}
ref_provider=${5:-trt}

if [ ! -f "$model" ]; then
    echo "$model not found."
    exit
fi

tmp_dir=$(mktemp -d)

PYTHON=$(which python3)
ONNX_BENCH=./build/onnx_bench

# prune model
prune_model=$tmp_dir/pruned.onnx
if [ -z "$node_names" ];then
    $PYTHON prune.py $model $prune_model
else
    $PYTHON prune.py $model $prune_model --output_nodes "$node_names"
fi

# run model to get output
base_out=$tmp_dir/${base_provider}_out.txt
ref_out=$tmp_dir/${ref_provider}_out.txt

# export ORT_DEBUG_NODE_IO_DUMP_SHAPE_DATA=1
# export ORT_DEBUG_NODE_IO_DUMP_NODE_PLACEMENT=1
# export ORT_DEBUG_NODE_IO_DUMP_INPUT_DATA=0
# export ORT_DEBUG_NODE_IO_DUMP_OUTPUT_DATA=1
# export ORT_DEBUG_NODE_IO_DUMP_DATA_DESTINATION=stdout

$ONNX_BENCH --onnx $prune_model --provider $base_provider --dumpOutput $base_out > cpu.txt
$ONNX_BENCH --onnx $prune_model --provider $ref_provider --precision $precision --dumpOutput $ref_out > trt.txt

# check precision
$PYTHON check.py $base_out $ref_out

rm -rf $tmp_dir