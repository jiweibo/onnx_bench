# onnx_bench

onnx benchmark and tools

- install deps

```
sudo apt install libgflags-dev
sudo apt install libgoogle-glog-dev
```



- compile

```
export ORT_ROOTDIR=../../onnxruntime/build/install/
cmake .. -DORT_ROOTDIR=$ORT_ROOTDIR \
         -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
```
