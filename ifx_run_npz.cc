
/*
* ./ifx_run_npz --ifxs "data_2/trajectory_classifier/cyclist_vectornet_3060_trt102.ifxmodel data_2/trajectory_classifier/pedestrian_vectornet_3060_trt102.ifxmodel data_2/trajectory_classifier/vectornet_3060_trt102.ifxmodel data_2/trajectory_classifier/pedestrian_vectornet_3060_trt102.ifxmodel data_2/trajectory_classifier/pedestrian_vectornet_3060_trt102.ifxmodel"  --npz_dirs "/wilber/workspace/cyc_npz /wilber/workspace/ped_npz/ /wilber/workspace/veh_npz/ /wilber/workspace/ped_npz/ /wilber/workspace/ped_npz/"
*/

#include <climits>
#include <cstddef>
#include <dirent.h>
#include <future>
#include <sys/stat.h>

#include "cnpy.h"
#include "ifx_sess.h"
#include "thread_pool.h"
#include "utils/barrier.h"

using namespace ifx_sess;

DEFINE_string(ifxs, "", "a.ifxmodel b.ifxmodel c.ifxmodel ....");
DEFINE_string(npz_dirs, "a_npz b_npz c_npz ...", "");

namespace {

// Helper function to split a string by a delimiter and return a vector of
// tokens
std::vector<std::string> SplitString(const std::string& str, char delimiter) {
  std::vector<std::string> tokens;
  std::string token;
  std::istringstream tokenStream(str);
  while (std::getline(tokenStream, token, delimiter)) {
    LOG(INFO) << token;
    tokens.push_back(token);
  }
  return tokens;
}

std::vector<SessConfig> ParseFlags(const std::vector<std::string>& ifx_vecs) {
  std::vector<SessConfig> configs(ifx_vecs.size());
  for (size_t i = 0; i < ifx_vecs.size(); ++i) {
    SessConfig config;
    config.device_id = 0;
    config.ifx_file = ifx_vecs[i];
    config.cache_dir = "";
    config.use_gpu = true;
    config.enable_fp16 = true;
    configs[i] = std::move(config);
  }
  return configs;
}

std::vector<std::string> ListFiles(const std::string& dir) {
  std::vector<std::string> files;
  DIR* dp = opendir(dir.c_str());
  if (dp == nullptr) {
    LOG(FATAL) << "Failed to open directory: " << dir;
  }
  struct dirent* entry;
  while ((entry = readdir(dp)) != nullptr) {
    std::string name = entry->d_name;
    if (name == "." || name == "..") {
      continue;
    }
    std::string fullPath = dir + "/" + name;
    struct stat statbuf;
    if (stat(fullPath.c_str(), &statbuf) != 0) {
      std::cerr << "Cannot stat file: " << fullPath << std::endl;
      continue;
    }
    if (S_ISREG(statbuf.st_mode)) {
      files.push_back(fullPath);
    }
  }
  closedir(dp);
  return files;
}

} // namespace

void RunNpzData(Ifx_Sess* sess, const std::string& npz_file, Barrier* barrier = nullptr) {
  cnpy::npz_t npz_data = cnpy::npz_load(npz_file);
  std::map<std::string, std::shared_ptr<core::Tensor>> in_tensors;
  for (size_t j = 0; j < sess->InputNames().size(); ++j) {
    auto& name = sess->InputNames()[j];
    auto in_dims = sess->InputDims()[j];
    auto ifx_tensor = std::make_shared<core::Tensor>(
        &(*npz_data.at(name).data_holder)[0], npz_data.at(name).num_bytes(), core::Dims(in_dims),
        ifx_sess::ToDataType(sess->InputDtypes()[j]), core::Location::kHOST);
    in_tensors.emplace(name, ifx_tensor);
  }
  if (barrier)
    barrier->Wait();
  sess->Run(in_tensors);
}

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  auto ifx_paths = SplitString(FLAGS_ifxs, ' ');
  auto npz_dirs = SplitString(FLAGS_npz_dirs, ' ');
  auto configs = ParseFlags(ifx_paths);
  ThreadPool tp(configs.size());
  Barrier barrier(configs.size());

  LOG(INFO) << "configs size " << configs.size();

  std::vector<std::unique_ptr<Ifx_Sess>> sessions;
  for (size_t i = 0; i < configs.size(); ++i) {
    std::unique_ptr<Ifx_Sess> tmp = std::make_unique<Ifx_Sess>(configs[i]);
    sessions.emplace_back(std::move(tmp));
  }
  LOG(INFO) << "Init session done";

  // List dir.
  size_t iter_num = UINT_MAX;
  std::vector<std::vector<std::string>> files;
  for (size_t i = 0; i < npz_dirs.size(); ++i) {
    auto file_vec = ListFiles(npz_dirs[i]);
    iter_num = std::min(iter_num, file_vec.size());
    files.emplace_back(std::move(file_vec));
  }
  LOG(INFO) << "iter_num is " << iter_num;

  Barrier* bar = &barrier;
  for (size_t i = 0; i < iter_num; ++i) {
    if (i % 100 == 0) {
      LOG(INFO) << "Iter " << i;
    }
    std::vector<std::future<void>> futures;
    for (size_t m = 0; m < configs.size(); ++m) {
      if (i > files[m].size()) {
        bar = nullptr;
        continue;
      }
      futures.emplace_back(tp.enqueue(
          [](Ifx_Sess* sess, const std::string& npz_file, Barrier* barrier) { RunNpzData(sess, npz_file, barrier); },
          sessions[m].get(), std::cref(files[m][i]), bar));
    }
    for (auto& f : futures)
      f.get();
  }

  return 0;
}