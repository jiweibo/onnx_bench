set -x
set -e

DIR="$(cd "$(dirname "$0")" ; pwd -P)"
cd "$DIR"

model=$1
precision=$2
node_names=$3
input_npz=$4
base_provider=${5:-cuda}
ref_provider=${6:-trt}

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
    $PYTHON prune.py $model $prune_model --output_nodes "$node_names" --prune
fi

# run model to get output
base_out=$tmp_dir/${base_provider}_out.npz
ref_out=$tmp_dir/${ref_provider}_out.npz

if [[ -z "$input_npz" ]];then
  load_input=""
else
  load_input="--loadInputs $input_npz"
fi

$ONNX_BENCH --onnx $prune_model --provider $base_provider --dumpOutput $base_out $load_input
$ONNX_BENCH --onnx $prune_model --provider $ref_provider --precision $precision --dumpOutput $ref_out $load_input

# check precision
$PYTHON check.py $base_out $ref_out

rm -rf $tmp_dir
