# onnx_bench

onnx benchmark and tools

- install deps

```
sudo apt install libgflags-dev
sudo apt install libgoogle-glog-dev
```


- compile

```
export ORT_DIR=../../onnxruntime/build/install/
cmake .. -DORT_DIR=$ORT_DIR \
         -DJSONCPP_DIR=$JSONCPP_DIR \
         -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
```


- onnx_bench

    -batch (batch) type: int32 default: 1
    -cacheDir (the cache dir) type: string default: ""
    -dumpOutput (Print the output tensor(s) of the last inference iteration
      (default = disabled).) type: string default: ""
    -inputType (txt, bin etc.) type: string default: "txt"
    -onnx (onnx model file) type: string default: ""
    -precision (fp32, fp16, int8) type: string default: "fp32"
    -provider (cpu, cuda, trt) type: string default: "cpu"
    -repeats (repeats) type: int32 default: 1
    -warmup (warmup) type: int32 default: 0

- check model precision

    ./check_prec.sh model_file [output_node_name (default all_model_outputs)] [base provider (default cpu)] [ref provider (default trt)]

    if output_node_name is a intermediate tensor of the model, it will prune the model first, and then run in the pruned model to get output.
