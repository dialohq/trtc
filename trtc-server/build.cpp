// The `build` subcommand: trtc_build_spec.json + ONNX -> TensorRT engines + manifest.json.
//
// All build options live in the spec sitting next to the ONNX — the entire
// tensorrt::IBuilderConfig, as JSON. C++ has no reflection, so the mapping is
// an explicit table derived from the NvInfer headers of every pinned TensorRT
// version; unknown names fail loudly with the list this TensorRT actually
// has. Runs on hardware matching the deployment GPU, against the one
// TensorRT this binary was built with.
//
//   trtc-server build <spec dir | model.onnx> [--out DIR] [--timing-cache FILE] [--force]
//
// A bare model.onnx uses the trtc_build_spec.json next to it, or defaults
// (strongly typed, TensorRT defaults, no profiles). The spec:
//
//   { "trtc_build_spec": 1,
//     "components": [ {
//       "onnx": "model.onnx",
//       "strongly_typed": true,
//       "profiles": [ {"x": {"min": [1,8], "opt": [4,8], "max": [16,8]}} ],
//       "builder_config": { "flags": ["TF32"],
//                           "memory_pool_limits": {"WORKSPACE": "4G"},
//                           "builder_optimization_level": 5, ... },
//       "onnx_sha256": "...",                     // optional, verified
//       "external_data": {"model.onnx.data": null} // optional, >2GB models
//     } ] }
//
// TRTC_CACHE_DIR points at a persistent cache; built engines are stored there
// by content key and reused when the same build is requested again.

#include <NvInfer.h>
#include <NvOnnxParser.h>

#include <chrono>
#include <map>
#include <memory>

#include "common.hpp"

using namespace trtc;

// ---- the IBuilderConfig vocabulary, one entry per option TensorRT has ----

template <typename Enum>
static Enum lookup(const std::map<std::string, Enum> &table, const std::string &name, const std::string &context) {
  auto it = table.find(name);
  if (it != table.end()) return it->second;
  std::string known;
  for (const auto &[key, _] : table) known += (known.empty() ? "" : ", ") + key;
  throw std::runtime_error(context + ": this TensorRT has no '" + name + "' (known: " + known + ")");
}

#define E(table, name) {#name, table::k##name}

static const std::map<std::string, nvinfer1::BuilderFlag> BUILDER_FLAGS = {
    E(nvinfer1::BuilderFlag, DEBUG),
    E(nvinfer1::BuilderFlag, DIRECT_IO),
    E(nvinfer1::BuilderFlag, DISABLE_COMPILATION_CACHE),
    E(nvinfer1::BuilderFlag, DISABLE_TIMING_CACHE),
    E(nvinfer1::BuilderFlag, DISTRIBUTIVE_INDEPENDENCE),
    E(nvinfer1::BuilderFlag, EDITABLE_TIMING_CACHE),
    E(nvinfer1::BuilderFlag, ERROR_ON_TIMING_CACHE_MISS),
    E(nvinfer1::BuilderFlag, EXCLUDE_LEAN_RUNTIME),
    E(nvinfer1::BuilderFlag, GPU_FALLBACK),
    E(nvinfer1::BuilderFlag, MONITOR_MEMORY),
    E(nvinfer1::BuilderFlag, REFIT),
    E(nvinfer1::BuilderFlag, REFIT_IDENTICAL),
    E(nvinfer1::BuilderFlag, REFIT_INDIVIDUAL),
    E(nvinfer1::BuilderFlag, SAFETY_SCOPE),
    E(nvinfer1::BuilderFlag, SPARSE_WEIGHTS),
    E(nvinfer1::BuilderFlag, STRICT_NANS),
    E(nvinfer1::BuilderFlag, STRIP_PLAN),
    E(nvinfer1::BuilderFlag, TF32),
    E(nvinfer1::BuilderFlag, VERSION_COMPATIBLE),
    E(nvinfer1::BuilderFlag, WEIGHT_STREAMING),
#if NV_TENSORRT_MAJOR < 11  // TensorRT 11 is strongly-typed only: precision flags are gone
    E(nvinfer1::BuilderFlag, BF16),
    E(nvinfer1::BuilderFlag, FP16),
    E(nvinfer1::BuilderFlag, FP4),
    E(nvinfer1::BuilderFlag, FP8),
    E(nvinfer1::BuilderFlag, INT4),
    E(nvinfer1::BuilderFlag, INT8),
    E(nvinfer1::BuilderFlag, OBEY_PRECISION_CONSTRAINTS),
    E(nvinfer1::BuilderFlag, PREFER_PRECISION_CONSTRAINTS),
    E(nvinfer1::BuilderFlag, REJECT_EMPTY_ALGORITHMS),
#endif
};

static const std::map<std::string, nvinfer1::MemoryPoolType> MEMORY_POOLS = {
    E(nvinfer1::MemoryPoolType, WORKSPACE),
    E(nvinfer1::MemoryPoolType, DLA_MANAGED_SRAM),
    E(nvinfer1::MemoryPoolType, DLA_LOCAL_DRAM),
    E(nvinfer1::MemoryPoolType, DLA_GLOBAL_DRAM),
    E(nvinfer1::MemoryPoolType, TACTIC_DRAM),
    E(nvinfer1::MemoryPoolType, TACTIC_SHARED_MEMORY),
};

static const std::map<std::string, nvinfer1::HardwareCompatibilityLevel> HW_COMPAT = {
    E(nvinfer1::HardwareCompatibilityLevel, NONE),
    E(nvinfer1::HardwareCompatibilityLevel, AMPERE_PLUS),
    E(nvinfer1::HardwareCompatibilityLevel, SAME_COMPUTE_CAPABILITY),
};

static const std::map<std::string, nvinfer1::EngineCapability> ENGINE_CAPABILITIES = {
    E(nvinfer1::EngineCapability, STANDARD),
    E(nvinfer1::EngineCapability, SAFETY),
    E(nvinfer1::EngineCapability, DLA_STANDALONE),
};

static const std::map<std::string, nvinfer1::DeviceType> DEVICE_TYPES = {
    E(nvinfer1::DeviceType, GPU),
    E(nvinfer1::DeviceType, DLA),
};

static const std::map<std::string, nvinfer1::ProfilingVerbosity> PROFILING_VERBOSITIES = {
    E(nvinfer1::ProfilingVerbosity, NONE),
    E(nvinfer1::ProfilingVerbosity, LAYER_NAMES_ONLY),
    E(nvinfer1::ProfilingVerbosity, DETAILED),
};

static const std::map<std::string, nvinfer1::TacticSource> TACTIC_SOURCES = {
    E(nvinfer1::TacticSource, EDGE_MASK_CONVOLUTIONS),
    E(nvinfer1::TacticSource, JIT_CONVOLUTIONS),
#if NV_TENSORRT_MAJOR < 11
    E(nvinfer1::TacticSource, CUBLAS),
    E(nvinfer1::TacticSource, CUBLAS_LT),
    E(nvinfer1::TacticSource, CUDNN),
#endif
};

static const std::map<std::string, nvinfer1::PreviewFeature> PREVIEW_FEATURES = {
    E(nvinfer1::PreviewFeature, ALIASED_PLUGIN_IO_10_03),
    E(nvinfer1::PreviewFeature, RUNTIME_ACTIVATION_RESIZE_10_10),
#if NV_TENSORRT_MAJOR < 11
    E(nvinfer1::PreviewFeature, PROFILE_SHARING_0806),
#endif
};

static const std::map<std::string, nvinfer1::RuntimePlatform> RUNTIME_PLATFORMS = {
    E(nvinfer1::RuntimePlatform, SAME_AS_BUILD),
    E(nvinfer1::RuntimePlatform, WINDOWS_AMD64),
};

static const std::map<std::string, nvinfer1::TilingOptimizationLevel> TILING_LEVELS = {
    E(nvinfer1::TilingOptimizationLevel, NONE),
    E(nvinfer1::TilingOptimizationLevel, FAST),
    E(nvinfer1::TilingOptimizationLevel, MODERATE),
    E(nvinfer1::TilingOptimizationLevel, FULL),
};

#undef E

// 4294967296, "4G", "512M" or "1.5G" -> bytes.
static int64_t parse_size(const json &value, const std::string &context) {
  if (value.is_number_integer()) return value;
  if (value.is_string()) {
    std::string raw = value;
    static const std::map<char, int64_t> suffixes = {{'K', 1LL << 10}, {'M', 1LL << 20}, {'G', 1LL << 30}, {'T', 1LL << 40}};
    auto it = raw.empty() ? suffixes.end() : suffixes.find(std::toupper(raw.back()));
    if (it != suffixes.end()) return int64_t(std::stod(raw.substr(0, raw.size() - 1)) * it->second);
    return std::stoll(raw);
  }
  throw std::runtime_error(context + ": expected bytes or a \"4G\"-style string");
}

static void apply_builder_config(const json &cfg, nvinfer1::IBuilderConfig &config) {
  std::string known =
      "flags, memory_pool_limits, builder_optimization_level, avg_timing_iterations, max_aux_streams, "
      "max_num_tactics, dla_core, default_device_type, engine_capability, hardware_compatibility_level, "
      "profiling_verbosity, runtime_platform, tiling_optimization_level, l2_limit_for_tiling, tactic_sources, "
      "preview_features, quantization_flags";
  for (const auto &[key, value] : cfg.items()) {
    if (key == "flags") {
      for (const std::string &name : value) config.setFlag(lookup(BUILDER_FLAGS, name, "builder_config.flags"));
    } else if (key == "memory_pool_limits") {
      for (const auto &[pool, limit] : value.items())
        config.setMemoryPoolLimit(lookup(MEMORY_POOLS, pool, "builder_config.memory_pool_limits"),
                                  parse_size(limit, "builder_config.memory_pool_limits." + pool));
    } else if (key == "builder_optimization_level") {
      config.setBuilderOptimizationLevel(value);
    } else if (key == "avg_timing_iterations") {
      config.setAvgTimingIterations(value);
    } else if (key == "max_aux_streams") {
      config.setMaxAuxStreams(value);
    } else if (key == "max_num_tactics") {
      config.setMaxNbTactics(value);
    } else if (key == "dla_core") {
      config.setDLACore(value);
    } else if (key == "default_device_type") {
      config.setDefaultDeviceType(lookup(DEVICE_TYPES, value, "builder_config.default_device_type"));
    } else if (key == "engine_capability") {
      config.setEngineCapability(lookup(ENGINE_CAPABILITIES, value, "builder_config.engine_capability"));
    } else if (key == "hardware_compatibility_level") {
      config.setHardwareCompatibilityLevel(lookup(HW_COMPAT, value, "builder_config.hardware_compatibility_level"));
    } else if (key == "profiling_verbosity") {
      config.setProfilingVerbosity(lookup(PROFILING_VERBOSITIES, value, "builder_config.profiling_verbosity"));
    } else if (key == "runtime_platform") {
      config.setRuntimePlatform(lookup(RUNTIME_PLATFORMS, value, "builder_config.runtime_platform"));
    } else if (key == "tiling_optimization_level") {
      config.setTilingOptimizationLevel(lookup(TILING_LEVELS, value, "builder_config.tiling_optimization_level"));
    } else if (key == "l2_limit_for_tiling") {
      config.setL2LimitForTiling(parse_size(value, "builder_config.l2_limit_for_tiling"));
    } else if (key == "tactic_sources") {
      uint32_t mask = 0;
      for (const std::string &name : value)
        mask |= 1u << uint32_t(lookup(TACTIC_SOURCES, name, "builder_config.tactic_sources"));
      config.setTacticSources(mask);
    } else if (key == "preview_features") {
      for (const auto &[name, enabled] : value.items())
        config.setPreviewFeature(lookup(PREVIEW_FEATURES, name, "builder_config.preview_features"), enabled);
    } else if (key == "quantization_flags") {
#if NV_TENSORRT_MAJOR < 11
      for (const std::string &name : value) {
        if (name != "CALIBRATE_BEFORE_FUSION")
          throw std::runtime_error("builder_config.quantization_flags: known: CALIBRATE_BEFORE_FUSION");
        config.setQuantizationFlag(nvinfer1::QuantizationFlag::kCALIBRATE_BEFORE_FUSION);
      }
#else
      throw std::runtime_error("builder_config.quantization_flags: gone in TensorRT 11");
#endif
    } else {
      throw std::runtime_error("builder_config." + key + ": no such option (known: " + known + ")");
    }
  }
}

// ---- spec plumbing ----

static const char *SPEC_FILE = "trtc_build_spec.json";

// A spec dir, or a bare .onnx (sibling spec if present, else defaults).
static std::pair<fs::path, json> load_spec(const fs::path &target) {
  if (fs::is_directory(target)) {
    fs::path spec_path = target / SPEC_FILE;
    if (!fs::exists(spec_path)) throw std::runtime_error("no " + std::string(SPEC_FILE) + " in " + target.string());
    return {target, read_json(spec_path)};
  }
  if (target.extension() != ".onnx" || !fs::exists(target))
    throw std::runtime_error("target must be a spec directory or an existing .onnx file: " + target.string());
  fs::path dir = fs::absolute(target).parent_path();
  if (fs::exists(dir / SPEC_FILE)) return {dir, read_json(dir / SPEC_FILE)};
  return {dir, {{"trtc_build_spec", 1}, {"components", json::array({{{"onnx", target.filename().string()}}})}}};
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

static nvinfer1::Dims to_dims(const json &values) {
  nvinfer1::Dims dims{};
  dims.nbDims = int(values.size());
  for (size_t i = 0; i < values.size(); ++i) dims.d[i] = values[i];
  return dims;
}

static void build_engine(const json &component, const fs::path &onnx_path, const fs::path &engine_file,
                         nvinfer1::ITimingCache *timing_cache, nvinfer1::IBuilder &builder, Logger &logger) {
  bool strongly_typed = component.value("strongly_typed", true);
  auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder.createNetworkV2(
      strongly_typed ? 1U << uint32_t(nvinfer1::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED) : 0));
  auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger));
  // parseFromFile, not bytes: >2GB models keep their weights in external data
  // files next to the ONNX, resolved relative to its path.
  if (!parser->parseFromFile(onnx_path.c_str(), int(nvinfer1::ILogger::Severity::kWARNING))) {
    std::string errors;
    for (int i = 0; i < parser->getNbErrors(); ++i) errors += std::string(parser->getError(i)->desc()) + "\n";
    throw std::runtime_error("TensorRT failed to parse " + onnx_path.string() + ":\n" + errors);
  }

  auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder.createBuilderConfig());
  apply_builder_config(component.value("builder_config", json::object()), *config);
  if (timing_cache) config->setTimingCache(*timing_cache, false);
  for (const json &profile_shapes : component.value("profiles", json::array())) {
    nvinfer1::IOptimizationProfile *profile = builder.createOptimizationProfile();
    for (const auto &[tensor, ranges] : profile_shapes.items()) {
      profile->setDimensions(tensor.c_str(), nvinfer1::OptProfileSelector::kMIN, to_dims(ranges.at("min")));
      profile->setDimensions(tensor.c_str(), nvinfer1::OptProfileSelector::kOPT, to_dims(ranges.at("opt")));
      profile->setDimensions(tensor.c_str(), nvinfer1::OptProfileSelector::kMAX, to_dims(ranges.at("max")));
    }
    config->addOptimizationProfile(profile);
  }
  auto serialized = std::unique_ptr<nvinfer1::IHostMemory>(builder.buildSerializedNetwork(*network, *config));
  if (!serialized) throw std::runtime_error("TensorRT failed to build engine from " + onnx_path.string());
  write_file(engine_file, std::string(static_cast<const char *>(serialized->data()), serialized->size()));
}

int build_main(int argc, char **argv) try {
  std::string target, out, timing_cache_path;
  bool force = false;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto next = [&] {
      if (++i >= argc) throw std::runtime_error("missing value for " + arg);
      return std::string(argv[i]);
    };
    if (arg == "--out") out = next();
    else if (arg == "--timing-cache") timing_cache_path = next();
    else if (arg == "--force") force = true;
    else if (arg.rfind("--", 0) == 0) throw std::runtime_error("unknown flag " + arg);
    else target = arg;
  }
  if (target.empty()) throw std::runtime_error("usage: trtc-server build <spec dir | model.onnx> [--out DIR] [--timing-cache FILE] [--force]");

  auto [work_dir, spec] = load_spec(target);
  if (spec.value("trtc_build_spec", 0) != 1)
    throw std::runtime_error("unsupported trtc_build_spec version (expected 1)");
  fs::path out_dir = out.empty() ? work_dir : fs::path(out);
  fs::create_directories(out_dir);

  json gpu = query_gpu();
  fs::path cache_dir = getenv("TRTC_CACHE_DIR") ? fs::path(getenv("TRTC_CACHE_DIR")) : fs::path();

  // One builder + timing cache shared across components.
  std::unique_ptr<nvinfer1::IBuilder> builder;
  std::unique_ptr<nvinfer1::IBuilderConfig> cache_holder;
  std::unique_ptr<nvinfer1::ITimingCache> timing_cache;
  Logger logger;
  auto ensure_builder = [&] {
    if (builder) return;
    // TensorRT aborts (not throws) without a CUDA device; fail cleanly first.
    if (gpu["compute_capability"].is_null())
      throw std::runtime_error("no CUDA device visible (is the driver injected?); engine builds need the deployment GPU");
    builder.reset(nvinfer1::createInferBuilder(logger));
    if (!timing_cache_path.empty()) {
      cache_holder.reset(builder->createBuilderConfig());
      std::string blob = fs::exists(timing_cache_path) ? read_file(timing_cache_path) : "";
      timing_cache.reset(cache_holder->createTimingCache(blob.data(), blob.size()));
    }
  };

  json built_components = json::array();
  for (const json &component : spec.at("components")) {
    std::string onnx_name = component.at("onnx");
    std::string stem = fs::path(onnx_name).stem();
    fs::path onnx_path = work_dir / onnx_name;

    // Verify every file the component references; declared hashes must match.
    std::map<std::string, std::string> file_hashes;
    json declared = {{onnx_name, component.value("onnx_sha256", json(nullptr))}};
    declared.update(component.value("external_data", json::object()));
    for (const auto &[file_name, declared_sha] : declared.items()) {
      fs::path file_path = work_dir / file_name;
      if (!fs::exists(file_path)) throw std::runtime_error("spec references missing file: " + file_path.string());
      std::string actual = sha256_file(file_path);
      if (declared_sha.is_string() && declared_sha != actual)
        throw std::runtime_error(file_path.string() + " does not match the spec: sha256 " + actual +
                                 ", spec says " + std::string(declared_sha));
      file_hashes[file_name] = actual;
    }

    // An engine is reusable only for the same file contents, build options,
    // TensorRT, and GPU arch.
    json component_for_key = component;
    component_for_key.erase("onnx_sha256");
    component_for_key.erase("external_data");
    std::string cache_key = sha256_hex(json{{"files", file_hashes},
                                            {"component", component_for_key},
                                            {"trt", installed_trt_version()},
                                            {"cc", gpu["compute_capability"]}}
                                           .dump());
    fs::path engine_file = out_dir / (stem + ".engine");
    fs::path key_file = out_dir / (stem + ".engine.key");
    fs::path cached_engine;
    if (!cache_dir.empty()) {
      fs::create_directories(cache_dir / "engines");
      cached_engine = cache_dir / "engines" / (cache_key + ".engine");
    }

    if (!force && fs::exists(engine_file) && fs::exists(key_file) && read_file(key_file) == cache_key) {
      std::printf("keep existing %s (%s)\n", engine_file.c_str(), cache_key.substr(0, 12).c_str());
    } else if (!force && !cached_engine.empty() && fs::exists(cached_engine)) {
      std::printf("cache hit %s (%s)\n", stem.c_str(), cache_key.substr(0, 12).c_str());
      fs::copy_file(cached_engine, engine_file, fs::copy_options::overwrite_existing);
      write_file(key_file, cache_key);
    } else {
      std::printf("build %s -> %s\n", stem.c_str(), engine_file.c_str());
      auto started = std::chrono::steady_clock::now();
      ensure_builder();
      build_engine(component, onnx_path, engine_file, timing_cache.get(), *builder, logger);
      write_file(key_file, cache_key);
      if (!cached_engine.empty()) fs::copy_file(engine_file, cached_engine, fs::copy_options::overwrite_existing);
      double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
      std::printf("built %s in %.1fs\n", stem.c_str(), seconds);
    }

    json built = component;
    built["engine"] = stem + ".engine";
    built["onnx_sha256"] = file_hashes[onnx_name];
    if (component.contains("external_data")) {
      json external = json::object();
      for (const auto &[file_name, _] : component.at("external_data").items()) external[file_name] = file_hashes[file_name];
      built["external_data"] = external;
    }
    built["engine_sha256"] = sha256_file(engine_file);
    built["engine_size"] = fs::file_size(engine_file);
    built_components.push_back(built);
  }

  if (timing_cache && !timing_cache_path.empty()) {
    auto blob = std::unique_ptr<nvinfer1::IHostMemory>(timing_cache->serialize());
    if (blob && blob->size())
      write_file(timing_cache_path, std::string(static_cast<const char *>(blob->data()), blob->size()));
  }

  // The manifest the client merges and the runtime validates: the spec plus
  // build facts.
  write_json(out_dir / "manifest.json",
             {{"trtc_build_spec", 1},
              {"components", built_components},
              {"build",
               {{"tensorrt_version", installed_trt_version()},
                {"gpu_name", gpu["gpu_name"]},
                {"compute_capability", gpu["compute_capability"]},
                {"driver_version", gpu["driver_version"]},
                {"used_timing_cache", bool(timing_cache)}}}});
  std::printf("engines + manifest.json written to %s\n", out_dir.c_str());
  return 0;
} catch (const std::exception &error) {
  std::fprintf(stderr, "trtc-server build: %s\n", error.what());
  return 1;
}
