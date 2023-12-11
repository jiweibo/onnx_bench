import os
from tqdm import tqdm
from pathlib import Path
import tarfile
import shutil
from enum import Enum

import metric_pb2
from google.protobuf import text_format


class Precision(Enum):
    FP32 = 0
    FP16 = 1
    FP16_INT8 = 2
    INT8 = 3


class RunVoyModel(object):
    def __init__(
        self, download_cmd, ort_cmd, download_dir, save_data_dir, save_cache_dir
    ):
        self.download_cmd = download_cmd
        self.ort_cmd = ort_cmd
        self.download_dir = download_dir
        self.save_data_dir = save_data_dir
        self.save_cache_dir = save_cache_dir
        self.runtime_metrics = "runtime.metrics"
        self.calibration_name = "calibration.flatbuffers"
        Path(download_dir).mkdir(exist_ok=True)
        if save_data_dir:
            Path(save_data_dir).mkdir(exist_ok=True)
        if save_cache_dir:
            Path(save_cache_dir).mkdir(exist_ok=True)

    def run(self, model_cmd, cache_cmd):
        model_path = self.download_model(model_cmd)
        cache_path, precision, batches, min_subgraph_size = self.download_cache(
            cache_cmd
        )
        self.exec_model(model_path, cache_path, precision, min_subgraph_size, batches)
        os.system('echo ""')

    # model_cmd: "perception.model-files refine_classifier_9type_gen3_new_dataset_attention_da_pse_full_55_iter_630000.onnx  1   ./lidar-classifier/cluster_refine_classifier_gen3.onnx       model files"
    def download_model(self, model_cmd):
        items = self.format_cmd(model_cmd)

        save_path = Path(self.download_dir).joinpath(items[3])
        save_dir = save_path.parent
        Path(save_dir).mkdir(exist_ok=True)

        if save_path.exists():
            return str(save_path)

        cmd = (
            self.download_cmd
            + " pull"
            + " "
            + items[0]
            + " "
            + items[1]
            + " -v "
            + items[2]
        )
        os.system(cmd)
        shutil.move(items[1], save_path)
        return str(save_path)

    # cache_cmd: "perception.model-files refine_classifier_9type_gen3_new_dataset_attention_da_pse_full_55_iter_630000.onnx_fp16_pg189.tgz 3  ./lidar-classifier/cluster_refine_classifier_gen3.onnx.pg189 model trt8 rebase cache"
    def download_cache(self, cache_cmd):
        items = self.format_cmd(cache_cmd)

        cmd = (
            self.download_cmd + " pull " + items[0] + " " + items[1] + " -v " + items[2]
        )
        os.system('echo "Download ' + items[1] + " and unzip to " + items[3] + '"')
        os.system(cmd)

        save_path = Path(self.download_dir).joinpath(items[3])

        if save_path.exists():
            shutil.rmtree(save_path)

        precision = Precision.FP32
        t = tarfile.open(items[1])
        for name in t.getnames():
            if name.endswith(".engine"):
                if "fp16_int8" in name:
                    precision = Precision.FP16_INT8
                elif "fp16" in name:
                    precision = Precision.FP16
                unzip_name = str(Path(name).parent)
                break

        t.extractall(path=".")
        shutil.move(unzip_name, save_path.name)
        shutil.move(save_path.name, save_path.parent)
        os.remove(items[1])
        metrics_file = save_path.joinpath(self.runtime_metrics)
        batches, min_subgraph_size = self.parse_metric(metrics_file)

        return str(save_path), precision, batches, min_subgraph_size

    def format_cmd(self, cmd):
        items = cmd.split(" ")
        for i in range(len(items) - 1, -1, -1):
            if items[i] == "":
                items.pop(i)
        return items

    def parse_metric(self, path):
        metric = metric_pb2.OnnxruntimeMetrics()
        with open(path, "r") as f:
            text_format.Parse(f.read(), metric)
        min_subgraph_size = metric.trt_min_subgraph_size
        batches = set()
        for m in metric.metrics:
            for k, v in m.inputs.items():
                batches.add(v.dim_sizes[0])
        if len(batches) == 1 and batches[0] != 1:
            batches = []
        return sorted(batches), min_subgraph_size

    def exec_model(
        self, model_path, cache_path, precision, trt_min_subgraph_size, batches
    ):
        cmd = [self.ort_cmd]
        cmd.append("--onnx " + model_path)
        cmd.append("--provider trt")
        if precision == Precision.FP32:
            cmd.append("--precision fp32")
        elif precision == Precision.FP16:
            cmd.append("--precision fp16")
        elif precision == Precision.FP16_INT8:
            cmd.append(
                "--precision fp16 --precisionInt8 --calibrationName "
                + self.calibration_name
            )
        if self.save_data_dir:
            cmd.append(
                "--dumpOutput "
                + str(Path(self.save_data_dir).joinpath(Path(model_path).name + ".npz"))
            )

        if self.save_cache_dir:
            save_cache_path = Path(self.save_cache_dir).joinpath(Path(cache_path).name)
            save_cache_path.mkdir(exist_ok=True)
            shutil.copy(
                Path(cache_path).joinpath(self.runtime_metrics),
                save_cache_path.joinpath(self.runtime_metrics),
            )
            if precision == Precision.FP16_INT8:
                shutil.copy(
                    Path(cache_path).joinpath(self.calibration_name),
                    save_cache_path.joinpath(self.calibration_name),
                )
            cmd.append("--cacheDir " + str(save_cache_path))
        else:
            cmd.append("--cacheDir " + cache_path)

        cmd.append("--minSubgraphSize " + str(trt_min_subgraph_size))

        basic_cmd = " ".join(cmd)
        if len(batches) == 0:
            os.system("echo " + basic_cmd)
            os.system(basic_cmd)
        else:
            for batch in batches:
                real_cmd = basic_cmd + " --batch " + str(batch)
                os.system("echo " + real_cmd)
                os.system(real_cmd)


model_fiels = [
    "perception.model-files lss_regnety800_dru_13_169000.onnx 1 ./lidar_detector_model/lidar_second_stage.onnx            model files",
]


model_caches = [
    "perception.model-files lss_regnety800_dru_13_169000.onnx_fp16_pg189.tgz 1 ./lidar_detector_model/lidar_second_stage.onnx.pg189",
]


if __name__ == "__main__":
    p = os.popen("which truck.py")
    download_cmd = p.read().strip()
    p.close()
    ort_cmd = "../../build/onnx_bench"
    download_dir = "download"
    save_data_dir = "data"
    save_cache_dir = None  # If None, will use the default cache path

    voy = RunVoyModel(
        download_cmd, ort_cmd, download_dir, save_data_dir, save_cache_dir
    )
    for model, cache in zip(model_fiels, model_caches):
        voy.run(model, cache)
