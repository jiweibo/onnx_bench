set -x
set -e

DIR="$(cd "$(dirname "$0")" ; pwd -P)"
cd "$DIR"

model=$1
node_names=$2
base_provider=${3:-cpu}
ref_provider=${4:-trt}

if [ ! -f "$model" ]; then
    echo "$model not found."
    exit
fi

tmp_dir=$(mktemp -d)

PYTHON=$(which python3)
ONNX_BENCH=./build/onnx_bench

# prune model
prune_model=$tmp_dir/pruned.onnx
$PYTHON prune.py $model $prune_model $node_names

# run model to get output
base_out=$tmp_dir/${base_provider}_out.txt
ref_out=$tmp_dir/${ref_provider}_out.txt
$ONNX_BENCH --onnx $prune_model --provider $base_provider --dumpOutput $base_out
$ONNX_BENCH --onnx $prune_model --provider $ref_provider --dumpOutput $ref_out

# check precision
$PYTHON check.py $base_out $ref_out