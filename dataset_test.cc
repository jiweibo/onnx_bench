
#include "dataset.h"


#include <cstdint>
#include <iostream>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <iterator>

DEFINE_string(ds, "", "dataset dir");
DEFINE_int32(batch, 1, "batch");
DEFINE_int32(trunk_size, 1, "trunk_size");
DEFINE_bool(drop, false,
            "If the remaining data is not enough for the batch, whether to "
            "discard it?");

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_ds == "") {
    LOG(FATAL) << "Please add --ds option";
  }

  JsonDataSet ds(FLAGS_ds, FLAGS_trunk_size);
  std::vector<std::string> names{"input_1", "output_1"};
  while (1) {
    auto map = ds.GetData(names, FLAGS_batch, FLAGS_drop);
    if (map.empty())
      break;

    void *data = std::get<0>(map[names[0]]);
    std::vector<int64_t> shape = std::get<1>(map[names[0]]);
    Dtype dtype = std::get<2>(map[names[0]]);
    size_t num = std::accumulate(shape.begin(), shape.end(), 1U,
                                 std::multiplies<size_t>());
    std::copy(shape.begin(), shape.end(), std::ostream_iterator<int64_t>(std::cout, " "));
    std::cout << std::endl;
    for (size_t i = 0; i < num; ++i) {
      LOG(INFO) << static_cast<float *>(data)[i];
    }
  }

  return 0;
}