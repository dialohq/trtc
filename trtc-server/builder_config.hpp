// The trtc_build_spec.json "builder_config" vocabulary: the entire
// tensorrt::IBuilderConfig, one table entry per option. C++ has no
// reflection, so these tables are derived from the NvInfer*.h headers of
// every pinned TensorRT version (see tensorrtPins in flake.nix), with
// version guards where TensorRT removed things. Unknown option or member
// names fail loudly with the list this TensorRT actually has.
#pragma once

#include <NvInfer.h>

#include <map>

#include "common.hpp"

namespace trtc {

template <typename Enum>
Enum lookup(const std::map<std::string, Enum> &table, const std::string &name, const std::string &context) {
  auto it = table.find(name);
  if (it != table.end()) return it->second;
  std::string known;
  for (const auto &[key, _] : table) known += (known.empty() ? "" : ", ") + key;
  throw std::runtime_error(context + ": this TensorRT has no '" + name + "' (known: " + known + ")");
}

#define E(table, name) {#name, table::k##name}

inline const std::map<std::string, nvinfer1::BuilderFlag> BUILDER_FLAGS = {
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

inline const std::map<std::string, nvinfer1::MemoryPoolType> MEMORY_POOLS = {
    E(nvinfer1::MemoryPoolType, WORKSPACE),
    E(nvinfer1::MemoryPoolType, DLA_MANAGED_SRAM),
    E(nvinfer1::MemoryPoolType, DLA_LOCAL_DRAM),
    E(nvinfer1::MemoryPoolType, DLA_GLOBAL_DRAM),
    E(nvinfer1::MemoryPoolType, TACTIC_DRAM),
    E(nvinfer1::MemoryPoolType, TACTIC_SHARED_MEMORY),
};

inline const std::map<std::string, nvinfer1::HardwareCompatibilityLevel> HW_COMPAT = {
    E(nvinfer1::HardwareCompatibilityLevel, NONE),
    E(nvinfer1::HardwareCompatibilityLevel, AMPERE_PLUS),
    E(nvinfer1::HardwareCompatibilityLevel, SAME_COMPUTE_CAPABILITY),
};

inline const std::map<std::string, nvinfer1::EngineCapability> ENGINE_CAPABILITIES = {
    E(nvinfer1::EngineCapability, STANDARD),
    E(nvinfer1::EngineCapability, SAFETY),
    E(nvinfer1::EngineCapability, DLA_STANDALONE),
};

inline const std::map<std::string, nvinfer1::DeviceType> DEVICE_TYPES = {
    E(nvinfer1::DeviceType, GPU),
    E(nvinfer1::DeviceType, DLA),
};

inline const std::map<std::string, nvinfer1::ProfilingVerbosity> PROFILING_VERBOSITIES = {
    E(nvinfer1::ProfilingVerbosity, NONE),
    E(nvinfer1::ProfilingVerbosity, LAYER_NAMES_ONLY),
    E(nvinfer1::ProfilingVerbosity, DETAILED),
};

inline const std::map<std::string, nvinfer1::TacticSource> TACTIC_SOURCES = {
    E(nvinfer1::TacticSource, EDGE_MASK_CONVOLUTIONS),
    E(nvinfer1::TacticSource, JIT_CONVOLUTIONS),
#if NV_TENSORRT_MAJOR < 11
    E(nvinfer1::TacticSource, CUBLAS),
    E(nvinfer1::TacticSource, CUBLAS_LT),
    E(nvinfer1::TacticSource, CUDNN),
#endif
};

inline const std::map<std::string, nvinfer1::PreviewFeature> PREVIEW_FEATURES = {
    E(nvinfer1::PreviewFeature, ALIASED_PLUGIN_IO_10_03),
    E(nvinfer1::PreviewFeature, RUNTIME_ACTIVATION_RESIZE_10_10),
#if NV_TENSORRT_MAJOR < 11
    E(nvinfer1::PreviewFeature, PROFILE_SHARING_0806),
#endif
};

inline const std::map<std::string, nvinfer1::RuntimePlatform> RUNTIME_PLATFORMS = {
    E(nvinfer1::RuntimePlatform, SAME_AS_BUILD),
    E(nvinfer1::RuntimePlatform, WINDOWS_AMD64),
};

inline const std::map<std::string, nvinfer1::TilingOptimizationLevel> TILING_LEVELS = {
    E(nvinfer1::TilingOptimizationLevel, NONE),
    E(nvinfer1::TilingOptimizationLevel, FAST),
    E(nvinfer1::TilingOptimizationLevel, MODERATE),
    E(nvinfer1::TilingOptimizationLevel, FULL),
};

#undef E

// The tables above as plain data — server.cpp embeds them into the served
// OpenAPI contract (/openapi.json) without touching TensorRT headers.
// Defined in build.cpp; a new table here must be added there too.
json builder_config_vocabulary();

// 4294967296, "4G", "512M" or "1.5G" -> bytes.
inline int64_t parse_size(const json &value, const std::string &context) {
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

inline void apply_builder_config(const json &cfg, nvinfer1::IBuilderConfig &config) {
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

}  // namespace trtc
