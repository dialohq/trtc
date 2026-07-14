// Small shared helpers for the trtc C++ builder: files, sha256, GPU facts.
#pragma once

#include <dlfcn.h>

#include <cctype>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <optional>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include <openssl/evp.h>

namespace trtc {

using json = nlohmann::json;  // std::map-backed: keys serialize sorted, like the client's sort_keys
namespace fs = std::filesystem;

inline std::string read_file(const fs::path &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot read " + path.string());
  std::ostringstream out;
  out << in.rdbuf();
  return out.str();
}

inline void write_file(const fs::path &path, const std::string &data) {
  fs::create_directories(path.parent_path());
  // Write-then-rename so concurrent readers never see a torn file (the
  // broker's status.json is polled while the worker updates it).
  fs::path tmp = path;
  tmp += ".tmp";
  {
    std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("cannot write " + tmp.string());
    out << data;
  }
  fs::rename(tmp, path);
}

inline void write_json(const fs::path &path, const json &payload) {
  write_file(path, payload.dump(2) + "\n");
}

inline json read_json(const fs::path &path) { return json::parse(read_file(path)); }

inline std::string sha256_hex(const std::string &data) {
  unsigned char digest[EVP_MAX_MD_SIZE];
  unsigned int length = 0;
  if (!EVP_Digest(data.data(), data.size(), digest, &length, EVP_sha256(), nullptr))
    throw std::runtime_error("sha256 failed");
  static const char hex[] = "0123456789abcdef";
  std::string out;
  for (unsigned int i = 0; i < length; ++i) {
    out += hex[digest[i] >> 4];
    out += hex[digest[i] & 0xf];
  }
  return out;
}

inline std::string sha256_file(const fs::path &path) { return sha256_hex(read_file(path)); }

// A component name becomes filesystem paths on the builder, so reject
// anything that isn't a single innocuous path segment (no separators, no
// '..', no leading dot) to prevent traversal.
inline bool is_safe_name(const std::string &name) {
  static const std::regex pattern(R"([A-Za-z0-9][A-Za-z0-9._-]*)");
  return std::regex_match(name, pattern) && name.find("..") == std::string::npos;
}

// Numeric dotted prefix of a TensorRT version, dropping local/build suffixes:
// '10.13.3.9.post1' and '10.13.3.9+cuda12' both -> {10, 13, 3, 9}.
inline std::vector<int> version_tuple(const std::string &version) {
  std::vector<int> parts;
  std::string numeric = version.substr(0, version.find('+'));
  std::stringstream stream(numeric);
  std::string part;
  while (std::getline(stream, part, '.')) {
    if (part.empty() || !std::all_of(part.begin(), part.end(), [](unsigned char c) { return std::isdigit(c); }))
      break;
    parts.push_back(std::stoi(part));
  }
  return parts;
}

// Driver build (e.g. '590.48.01') from /proc/driver/nvidia/version. The NVRM
// line's wording varies (proprietary vs open kernel module), so match the
// version number itself.
inline json nvidia_kernel_module_version() {
  std::ifstream proc("/proc/driver/nvidia/version");
  std::string line;
  while (std::getline(proc, line)) {
    if (line.rfind("NVRM", 0) != 0) continue;
    static const std::regex pattern(R"(\b(\d+\.\d+(?:\.\d+)*)\b)");
    std::smatch match;
    if (std::regex_search(line, match, pattern)) return match[1].str();
    return nullptr;
  }
  return nullptr;
}

// Hardware facts straight from the CUDA driver API, via dlopen on
// libcuda.so.1 — the host-injected driver library. Degrades to nulls off-GPU.
inline json query_gpu() {
  json info = {{"gpu_name", nullptr}, {"compute_capability", nullptr}, {"driver_version", nullptr}};

  if (void *cuda = dlopen("libcuda.so.1", RTLD_LAZY)) {
    auto cuInit = reinterpret_cast<int (*)(unsigned)>(dlsym(cuda, "cuInit"));
    auto cuDeviceGet = reinterpret_cast<int (*)(int *, int)>(dlsym(cuda, "cuDeviceGet"));
    auto cuDeviceGetName = reinterpret_cast<int (*)(char *, int, int)>(dlsym(cuda, "cuDeviceGetName"));
    auto cuDeviceGetAttribute = reinterpret_cast<int (*)(int *, int, int)>(dlsym(cuda, "cuDeviceGetAttribute"));
    int device = 0;
    if (cuInit && cuDeviceGet && cuInit(0) == 0 && cuDeviceGet(&device, 0) == 0) {
      char name[96] = {};
      if (cuDeviceGetName && cuDeviceGetName(name, sizeof(name), device) == 0) info["gpu_name"] = name;
      // CUdevice_attribute enums; part of the stable driver ABI.
      int major = 0, minor = 0;
      if (cuDeviceGetAttribute && cuDeviceGetAttribute(&major, 75, device) == 0 &&
          cuDeviceGetAttribute(&minor, 76, device) == 0)
        info["compute_capability"] = std::to_string(major) + "." + std::to_string(minor);
    }
  }
  info["driver_version"] = nvidia_kernel_module_version();
  return info;
}

}  // namespace trtc
