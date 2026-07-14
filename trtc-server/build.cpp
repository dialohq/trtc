// trtc-build: one bare ONNX + flags -> one TensorRT engine + manifest.json.
//
// Runs on hardware matching the deployment GPU, against the one TensorRT this
// binary was built with. The manifest it writes is the same schema the Python
// client assembles and the runtime validates (see trtc/plan.py).
//
//   trtc-build model.onnx --name m --dtype float32 --workspace-gb 4 \
//       --trt-version 10.13.3.9 --out ./engines --timing-cache cache.bin \
//       [--shape NAME=MIN:OPT:MAX]... [--force]
//
// TRTC_CACHE_DIR points at a persistent cache; built engines are stored there
// by content key and reused when the same build is requested again.

#include <NvInfer.h>
#include <NvOnnxParser.h>

#include <chrono>
#include <cstring>
#include <memory>

#include "common.hpp"

using namespace trtc;

struct Shape {
  std::string name;
  std::vector<int64_t> min, opt, max;
};

// 'x=1x80:8x80:16x80' -> Shape; loud failure on anything malformed.
static Shape parse_shape(const std::string &spec) {
  auto fail = [&] { throw std::runtime_error("--shape expects NAME=MIN:OPT:MAX with 'x'-separated dims, got " + spec); };
  auto eq = spec.find('=');
  if (eq == std::string::npos || eq == 0) fail();
  Shape shape{spec.substr(0, eq), {}, {}, {}};
  std::vector<std::vector<int64_t> *> parts = {&shape.min, &shape.opt, &shape.max};
  std::stringstream ranges(spec.substr(eq + 1));
  std::string part;
  size_t index = 0;
  while (std::getline(ranges, part, ':')) {
    if (index >= 3) fail();
    std::stringstream dims(part);
    std::string dim;
    while (std::getline(dims, dim, 'x')) parts[index]->push_back(std::stoll(dim));
    ++index;
  }
  if (index != 3 || shape.min.size() != shape.opt.size() || shape.min.size() != shape.max.size() || shape.min.empty())
    fail();
  return shape;
}

static nvinfer1::Dims to_dims(const std::vector<int64_t> &values) {
  nvinfer1::Dims dims{};
  dims.nbDims = int(values.size());
  for (size_t i = 0; i < values.size(); ++i) dims.d[i] = values[i];
  return dims;
}

struct Logger : nvinfer1::ILogger {
  void log(Severity severity, const char *message) noexcept override {
    if (severity <= Severity::kWARNING) std::fprintf(stderr, "[trt] %s\n", message);
  }
};

static std::string installed_trt_version() {
  int v = getInferLibVersion();  // major * 10000 + minor * 100 + patch
  return std::to_string(v / 10000) + "." + std::to_string(v / 100 % 100) + "." + std::to_string(v % 100);
}

// Full numeric comparison on major.minor.patch: a '.postN' or build suffix on
// the pin is ignored, but a different patch is rejected. The environment must
// be correct; there is no override.
static void check_trt_version(const std::string &pinned) {
  auto pin = version_tuple(pinned);
  auto lib = version_tuple(installed_trt_version());
  pin.resize(3, 0);
  if (pin == lib) return;
  throw std::runtime_error("this builder has tensorrt " + installed_trt_version() + " but the plan pins " + pinned +
                           "; use a builder image built for that version instead");
}

int main(int argc, char **argv) try {
  std::string onnx, name, dtype = "float32", trt_version, out, timing_cache_path;
  double workspace_gb = 4.0;
  bool force = false;
  std::vector<Shape> shapes;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto next = [&] {
      if (++i >= argc) throw std::runtime_error("missing value for " + arg);
      return std::string(argv[i]);
    };
    if (arg == "--name") name = next();
    else if (arg == "--dtype") dtype = next();
    else if (arg == "--workspace-gb") workspace_gb = std::stod(next());
    else if (arg == "--trt-version") trt_version = next();
    else if (arg == "--out") out = next();
    else if (arg == "--timing-cache") timing_cache_path = next();
    else if (arg == "--shape") shapes.push_back(parse_shape(next()));
    else if (arg == "--force") force = true;
    else if (arg.rfind("--", 0) == 0) throw std::runtime_error("unknown flag " + arg);
    else onnx = arg;
  }
  if (onnx.empty()) throw std::runtime_error("usage: trtc-build model.onnx [flags]");
  fs::path onnx_path = fs::absolute(onnx);
  if (!fs::exists(onnx_path)) throw std::runtime_error("ONNX file not found: " + onnx_path.string());
  if (name.empty()) name = onnx_path.stem();
  if (!trt_version.empty()) check_trt_version(trt_version);
  fs::path out_dir = out.empty() ? onnx_path.parent_path() : fs::path(out);
  fs::create_directories(out_dir);

  json profiles = json::object();
  for (const auto &shape : shapes) profiles[shape.name] = {{"min", shape.min}, {"opt", shape.opt}, {"max", shape.max}};

  json gpu = query_gpu();
  int64_t workspace_bytes = int64_t(workspace_gb * (1LL << 30));
  std::string onnx_sha = sha256_file(onnx_path);

  // Same identity the Python builder used: an engine is reusable only for the
  // same ONNX bytes, profiles, dtype, workspace, TRT, and GPU arch.
  std::string cache_key = sha256_hex(json{{"onnx", onnx_sha},
                                          {"profiles", profiles},
                                          {"dtype", dtype},
                                          {"workspace", workspace_bytes},
                                          {"strongly_typed", true},
                                          {"trt", installed_trt_version()},
                                          {"cc", gpu["compute_capability"]}}
                                         .dump());
  fs::path engine_file = out_dir / (name + ".engine");
  fs::path key_file = out_dir / (name + ".engine.key");
  fs::path cached_engine;
  if (const char *cache_dir = getenv("TRTC_CACHE_DIR")) {
    fs::create_directories(fs::path(cache_dir) / "engines");
    cached_engine = fs::path(cache_dir) / "engines" / (cache_key + ".engine");
  }

  bool used_timing_cache = !timing_cache_path.empty();
  if (!force && fs::exists(engine_file) && fs::exists(key_file) && read_file(key_file) == cache_key) {
    std::printf("keep existing %s (%s)\n", engine_file.c_str(), cache_key.substr(0, 12).c_str());
  } else if (!force && !cached_engine.empty() && fs::exists(cached_engine)) {
    std::printf("cache hit %s (%s)\n", name.c_str(), cache_key.substr(0, 12).c_str());
    fs::copy_file(cached_engine, engine_file, fs::copy_options::overwrite_existing);
    write_file(key_file, cache_key);
  } else {
    // TensorRT aborts (not throws) without a CUDA device; fail cleanly first.
    if (gpu["compute_capability"].is_null())
      throw std::runtime_error("no CUDA device visible (is the driver injected?); engine builds need the deployment GPU");
    std::printf("build %s -> %s\n", name.c_str(), engine_file.c_str());
    auto started = std::chrono::steady_clock::now();

    Logger logger;
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(
        1U << uint32_t(nvinfer1::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED)));
    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger));
    if (!parser->parseFromFile(onnx_path.c_str(), int(nvinfer1::ILogger::Severity::kWARNING))) {
      std::string errors;
      for (int i = 0; i < parser->getNbErrors(); ++i) errors += std::string(parser->getError(i)->desc()) + "\n";
      throw std::runtime_error("TensorRT failed to parse " + onnx_path.string() + ":\n" + errors);
    }

    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, workspace_bytes);
    std::unique_ptr<nvinfer1::ITimingCache> timing_cache;
    if (used_timing_cache) {
      std::string blob = fs::exists(timing_cache_path) ? read_file(timing_cache_path) : "";
      timing_cache.reset(config->createTimingCache(blob.data(), blob.size()));
      config->setTimingCache(*timing_cache, false);
    }
    if (!shapes.empty()) {
      nvinfer1::IOptimizationProfile *profile = builder->createOptimizationProfile();
      for (const auto &shape : shapes) {
        profile->setDimensions(shape.name.c_str(), nvinfer1::OptProfileSelector::kMIN, to_dims(shape.min));
        profile->setDimensions(shape.name.c_str(), nvinfer1::OptProfileSelector::kOPT, to_dims(shape.opt));
        profile->setDimensions(shape.name.c_str(), nvinfer1::OptProfileSelector::kMAX, to_dims(shape.max));
      }
      config->addOptimizationProfile(profile);
    }

    auto serialized = std::unique_ptr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
    if (!serialized) throw std::runtime_error("TensorRT failed to build engine from " + onnx_path.string());
    write_file(engine_file, std::string(static_cast<const char *>(serialized->data()), serialized->size()));
    write_file(key_file, cache_key);
    if (!cached_engine.empty()) fs::copy_file(engine_file, cached_engine, fs::copy_options::overwrite_existing);
    if (timing_cache) {
      auto blob = std::unique_ptr<nvinfer1::IHostMemory>(timing_cache->serialize());
      if (blob && blob->size())
        write_file(timing_cache_path, std::string(static_cast<const char *>(blob->data()), blob->size()));
    }
    double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    std::printf("built %s in %.1fs\n", name.c_str(), seconds);
  }

  // The single-component manifest the client merges and the runtime validates
  // — same schema as the Python plan/manifest contract.
  json manifest = {
      {"trtc_plan", 1},
      {"bundle", name},
      {"tensorrt_version", installed_trt_version()},
      {"engine_dir_hint", nullptr},
      {"meta", json::object()},
      {"provenance", json::object()},
      {"components",
       json::array({{
           {"name", name},
           {"onnx", onnx_path.filename().string()},
           {"engine", name + ".engine"},
           {"dtype", dtype},
           {"workspace_bytes", workspace_bytes},
           {"strongly_typed", true},
           {"profiles", profiles},
           {"onnx_sha256", onnx_sha},
           {"meta", json::object()},
           {"engine_sha256", sha256_file(engine_file)},
           {"engine_size", fs::file_size(engine_file)},
       }})},
      {"build",
       {{"tensorrt_version", installed_trt_version()},
        {"gpu_name", gpu["gpu_name"]},
        {"compute_capability", gpu["compute_capability"]},
        {"driver_version", gpu["driver_version"]},
        {"used_timing_cache", used_timing_cache}}},
  };
  write_json(out_dir / "manifest.json", manifest);
  std::printf("engines + manifest.json written to %s\n", out_dir.c_str());
  return 0;
} catch (const std::exception &error) {
  std::fprintf(stderr, "trtc-build: %s\n", error.what());
  return 1;
}
