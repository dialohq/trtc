// trtc-server: the builder — a deliberately dumb HTTP job broker on GPU hardware.
//
// One job = one tar (trtc_build_spec.json + the ONNX and any external data
// files it references, side by side — the exact on-disk layout) = one engine
// back. All build options live in the spec; the request carries none. The
// server knows nothing about models or TensorRT — each job runs the
// trtc-build binary sitting next to this one, and a spec asking for options
// this image's TensorRT does not have fails the job loudly. Engine and
// timing caches persist under TRTC_DATA_DIR, so a stopped-and-resumed
// instance stays warm.
//
// API (bearer auth when TRTC_TOKEN is set):
//   POST /builds[?output_url=<presigned PUT for the engine>]
//        body: a job tar, or JSON {"input_url": ...} pointing at one
//   GET  /builds/{id}[?log_offset=N]   status + log; "result" holds the build
//        facts (single-component manifest) once succeeded
//   GET  /builds/{id}/artifacts        the engine, as raw bytes
//   GET  /info                         GPU facts, versions, cache location
//
// Environment:
//   TRTC_TOKEN             optional bearer token required on every request
//   TRTC_DATA_DIR          jobs + caches root (default ~/.cache/trtc)
//   TRTC_IDLE_TIMEOUT      seconds of inactivity before the server exits (0 = never)
//   TRTC_TENSORRT_VERSION  the TensorRT baked into this image (reported by /info)
//   TRTC_BUILD_EXE         build command override (default: trtc-build next to this binary)

#define CPPHTTPLIB_OPENSSL_SUPPORT
#include <httplib.h>

#include <archive.h>
#include <archive_entry.h>
#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <deque>
#include <mutex>
#include <random>
#include <thread>

#include "common.hpp"

using namespace trtc;

#ifndef TRTC_VERSION
#define TRTC_VERSION "dev"
#endif

static const char *SPEC_FILE = "trtc_build_spec.json";

static long now_seconds() {
  return std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

static double unix_time() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())
             .count() /
         1000.0;
}

struct State {
  fs::path jobs_dir, cache_dir;
  std::string build_exe;

  std::mutex lock;  // guards status files and the queue
  std::deque<std::string> queue;
  std::condition_variable wake;
  std::atomic<long> last_activity{now_seconds()};
  std::atomic<int> in_flight{0};

  void touch() { last_activity = now_seconds(); }
  fs::path job_dir(const std::string &id) { return jobs_dir / id; }

  std::optional<json> read_status(const std::string &id) {
    auto path = job_dir(id) / "status.json";
    if (!fs::exists(path)) return std::nullopt;
    return read_json(path);
  }

  json write_status(const std::string &id, const json &updates) {
    std::lock_guard guard(lock);
    json status = {{"id", id}};
    auto path = job_dir(id) / "status.json";
    if (fs::exists(path)) status = read_json(path);
    status.update(updates);
    write_json(path, status);
    return status;
  }
};

// Unpack one job tar into dest and return its validated spec. Members are
// written by hand (never a library extract-to-disk) and must be regular
// files with safe single-segment names; the spec must reference exactly one
// component whose files are all present.
static json extract_job_tar(const std::string &data, const fs::path &dest) {
  fs::create_directories(dest);
  auto archive = std::unique_ptr<struct archive, decltype(&archive_read_free)>(archive_read_new(), archive_read_free);
  archive_read_support_format_tar(archive.get());
  archive_read_support_filter_all(archive.get());  // plain, gzip, zstd — tar is tar
  if (archive_read_open_memory(archive.get(), data.data(), data.size()) != ARCHIVE_OK)
    throw std::runtime_error("not a tar: " + std::string(archive_error_string(archive.get())));

  struct archive_entry *entry;
  while (archive_read_next_header(archive.get(), &entry) == ARCHIVE_OK) {
    std::string name = archive_entry_pathname(entry) ? archive_entry_pathname(entry) : "";
    if (name.rfind("./", 0) == 0) name = name.substr(2);
    if (name.empty() || name.back() == '/') continue;  // directory entries
    if (archive_entry_filetype(entry) != AE_IFREG)
      throw std::runtime_error("job tar member '" + name + "' is not a regular file");
    if (!is_safe_name(name))
      throw std::runtime_error("job tar member '" + name + "' is not a safe file name");
    std::string content;
    const void *block;
    size_t size;
    la_int64_t offset;
    int code;
    while ((code = archive_read_data_block(archive.get(), &block, &size, &offset)) == ARCHIVE_OK)
      content.append(static_cast<const char *>(block), size);
    if (code != ARCHIVE_EOF) throw std::runtime_error("job tar read failed at '" + name + "'");
    write_file(dest / name, content);
  }

  if (!fs::exists(dest / SPEC_FILE)) throw std::runtime_error("job tar has no " + std::string(SPEC_FILE));
  json spec = read_json(dest / SPEC_FILE);
  if (spec.value("trtc_build_spec", 0) != 1)
    throw std::runtime_error("unsupported trtc_build_spec version (expected 1)");
  if (!spec.contains("components") || spec["components"].size() != 1)
    throw std::runtime_error("a builder job takes exactly one component");
  const json &component = spec["components"][0];
  json files = {{std::string(component.at("onnx")), nullptr}};
  files.update(component.value("external_data", json::object()));
  for (const auto &[name, _] : files.items()) {
    if (!is_safe_name(name)) throw std::runtime_error("spec references unsafe file name '" + name + "'");
    if (!fs::exists(dest / name))
      throw std::runtime_error("spec references '" + name + "' but the tar does not contain it");
  }
  return spec;
}

// Runs the trtc-build binary shipped next to this server on the extracted
// job dir; every build option comes from the spec inside it.
static std::vector<std::string> build_command(State &state, const fs::path &input_dir, const fs::path &out) {
  std::string trt = getenv("TRTC_TENSORRT_VERSION") ? getenv("TRTC_TENSORRT_VERSION") : "local";
  return {
      state.build_exe, input_dir.string(),
      "--out", out.string(),
      "--timing-cache", (state.cache_dir / ("timing_trt" + trt + ".bin")).string(),
  };
}

// fork/exec (no shell) with stdout+stderr appended to the job log.
static int run_logged(const std::vector<std::string> &argv, const fs::path &log_path, const fs::path &cache_dir) {
  std::string line = "$";
  for (const auto &arg : argv) line += " " + arg;
  std::ofstream(log_path, std::ios::app) << line << "\n";

  pid_t pid = fork();
  if (pid < 0) throw std::runtime_error("fork failed");
  if (pid == 0) {
    int fd = open(log_path.c_str(), O_WRONLY | O_APPEND | O_CREAT, 0644);
    dup2(fd, 1);
    dup2(fd, 2);
    setenv("TRTC_CACHE_DIR", cache_dir.c_str(), 1);
    std::vector<char *> raw;
    for (const auto &arg : argv) raw.push_back(const_cast<char *>(arg.c_str()));
    raw.push_back(nullptr);
    execvp(raw[0], raw.data());
    perror("execvp");
    _exit(127);
  }
  int wstatus = 0;
  waitpid(pid, &wstatus, 0);
  return WIFEXITED(wstatus) ? WEXITSTATUS(wstatus) : 128 + WTERMSIG(wstatus);
}

// "https://host[:port]/path?query" -> client base + path for httplib.
static std::pair<std::string, std::string> split_url(const std::string &url) {
  auto scheme_end = url.find("://");
  if (scheme_end == std::string::npos) throw std::runtime_error("unsupported url: " + url);
  auto path_start = url.find('/', scheme_end + 3);
  if (path_start == std::string::npos) return {url, "/"};
  return {url.substr(0, path_start), url.substr(path_start)};
}

static fs::path engine_path(State &state, const json &status) {
  fs::path onnx = std::string(status["spec"]["components"][0]["onnx"]);
  return state.job_dir(status["id"]) / "engines" / (onnx.stem().string() + ".engine");
}

static void run_job(State &state, const std::string &id) {
  fs::path job_dir = state.job_dir(id);
  fs::path input_dir = job_dir / "input";
  fs::path output_dir = job_dir / "engines";
  fs::path log_path = job_dir / "job.log";
  json status = state.read_status(id).value();
  state.in_flight++;
  state.touch();
  state.write_status(id, {{"state", "running"}, {"started_at", unix_time()}});
  try {
    if (status.contains("input_url") && !status["input_url"].is_null()) {
      auto [base, path] = split_url(status["input_url"]);
      httplib::Client client(base);
      client.set_read_timeout(600);
      client.set_follow_location(true);
      auto response = client.Get(path);
      if (!response || response->status != 200)
        throw std::runtime_error("input_url fetch failed (" + std::to_string(response ? response->status : 0) + ")");
      status = state.write_status(id, {{"spec", extract_job_tar(response->body, input_dir)}});
    }

    fs::create_directories(output_dir);
    int code = run_logged(build_command(state, input_dir, output_dir), log_path, state.cache_dir);
    if (code != 0) throw std::runtime_error("build command exited with " + std::to_string(code) + " (see job log)");

    if (status.contains("output_url") && !status["output_url"].is_null()) {
      auto [base, path] = split_url(status["output_url"]);
      httplib::Client client(base);
      client.set_write_timeout(600);
      auto response = client.Put(path, read_file(engine_path(state, status)), "application/octet-stream");
      if (!response || (response->status != 200 && response->status != 201 && response->status != 204))
        throw std::runtime_error("engine upload returned " + std::to_string(response ? response->status : 0));
    }

    fs::path manifest = output_dir / "manifest.json";
    json result = fs::exists(manifest) ? read_json(manifest) : json(nullptr);
    state.write_status(id, {{"state", "succeeded"}, {"finished_at", unix_time()}, {"result", result}});
  } catch (const std::exception &error) {
    std::ofstream(log_path, std::ios::app) << "ERROR: " << error.what() << "\n";
    state.write_status(id, {{"state", "failed"}, {"error", error.what()}, {"finished_at", unix_time()}});
  }
  state.in_flight--;
  state.touch();
}

static void worker_loop(State &state) {
  for (;;) {
    std::string id;
    {
      std::unique_lock guard(state.lock);
      state.wake.wait(guard, [&] { return !state.queue.empty(); });
      id = state.queue.front();
      state.queue.pop_front();
    }
    run_job(state, id);
  }
}

static std::string new_job_id() {
  static const char hex[] = "0123456789abcdef";
  std::random_device device;
  std::string id;
  for (int i = 0; i < 12; ++i) id += hex[device() % 16];
  return id;
}

static void send_json(httplib::Response &response, int code, const json &payload) {
  response.status = code;
  response.set_content(payload.dump(), "application/json");
}

int main(int argc, char **argv) {
  std::string host = "0.0.0.0";
  int port = 8080;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "serve") continue;  // `trtc-server serve` and `trtc-server` are the same thing
    if (arg == "--host" && i + 1 < argc) host = argv[++i];
    else if (arg == "--port" && i + 1 < argc) port = std::stoi(argv[++i]);
    else {
      std::fprintf(stderr, "usage: trtc-server [serve] [--host H] [--port P]\n");
      return 2;
    }
  }

  const char *home = getenv("HOME");
  fs::path data_dir = getenv("TRTC_DATA_DIR") ? fs::path(getenv("TRTC_DATA_DIR"))
                                              : fs::path(home ? home : "/") / ".cache" / "trtc";
  State state;
  state.jobs_dir = data_dir / "jobs";
  state.cache_dir = data_dir / "cache";
  fs::create_directories(state.jobs_dir);
  fs::create_directories(state.cache_dir);
  // The build tool ships next to this binary; TRTC_BUILD_EXE overrides (tests
  // stub it with a fake that needs no GPU).
  state.build_exe = getenv("TRTC_BUILD_EXE")
                        ? getenv("TRTC_BUILD_EXE")
                        : (fs::read_symlink("/proc/self/exe").parent_path() / "trtc-build").string();
  std::string token = getenv("TRTC_TOKEN") ? getenv("TRTC_TOKEN") : "";

  std::thread(worker_loop, std::ref(state)).detach();

  double idle_timeout = getenv("TRTC_IDLE_TIMEOUT") ? std::stod(getenv("TRTC_IDLE_TIMEOUT")) : 0;
  if (idle_timeout > 0)
    std::thread([&state, idle_timeout] {
      for (;;) {
        std::this_thread::sleep_for(std::chrono::seconds(30));
        // A job is dequeued before it runs, so also require no in-flight build.
        bool idle;
        {
          std::lock_guard guard(state.lock);
          idle = state.queue.empty() && state.in_flight == 0;
        }
        if (idle && now_seconds() - state.last_activity > idle_timeout) {
          std::fprintf(stderr, "idle for %.0fs, shutting down\n", idle_timeout);
          std::exit(0);
        }
      }
    }).detach();

  httplib::Server server;

  server.set_pre_routing_handler([&](const httplib::Request &request, httplib::Response &response) {
    state.touch();
    if (token.empty() || request.get_header_value("Authorization") == "Bearer " + token)
      return httplib::Server::HandlerResponse::Unhandled;
    send_json(response, 401, {{"error", "unauthorized"}});
    return httplib::Server::HandlerResponse::Handled;
  });

  server.Get("/info", [&](const httplib::Request &, httplib::Response &response) {
    size_t jobs = 0;
    for (auto it = fs::directory_iterator(state.jobs_dir); it != fs::directory_iterator(); ++it)
      if (it->is_directory()) ++jobs;
    json info = query_gpu();
    info["trtc"] = TRTC_VERSION;
    info["tensorrt"] = getenv("TRTC_TENSORRT_VERSION") ? json(getenv("TRTC_TENSORRT_VERSION")) : json(nullptr);
    info["jobs"] = jobs;
    info["cache_dir"] = state.cache_dir.string();
    send_json(response, 200, info);
  });

  server.Get(R"(/builds/([0-9a-f]+)/artifacts)", [&](const httplib::Request &request, httplib::Response &response) {
    auto status = state.read_status(request.matches[1]);
    if (!status) return send_json(response, 404, {{"error", "unknown job " + request.matches[1].str()}});
    fs::path engine = engine_path(state, *status);
    if ((*status)["state"] != "succeeded" || !fs::exists(engine))
      return send_json(response, 409,
                       {{"error", "job has no engine (state=" + std::string((*status)["state"]) + ")"}});
    response.set_content(read_file(engine), "application/octet-stream");
  });

  server.Get(R"(/builds/([0-9a-f]+))", [&](const httplib::Request &request, httplib::Response &response) {
    std::string id = request.matches[1];
    auto status = state.read_status(id);
    if (!status) return send_json(response, 404, {{"error", "unknown job " + id}});
    size_t offset = request.has_param("log_offset") ? std::stoul(request.get_param_value("log_offset")) : 0;
    json reply = *status;
    reply["log"] = "";
    reply["log_offset"] = offset;
    fs::path log_path = state.job_dir(id) / "job.log";
    if (fs::exists(log_path)) {
      std::string data = read_file(log_path);
      reply["log"] = offset < data.size() ? data.substr(offset) : "";
      reply["log_offset"] = data.size();
    }
    send_json(response, 200, reply);
  });

  server.Post("/builds", [&](const httplib::Request &request, httplib::Response &response) {
    std::string id = new_job_id();
    json status = {{"state", "queued"}, {"created_at", unix_time()}};
    try {
      if (request.has_param("output_url")) status["output_url"] = request.get_param_value("output_url");
      if (request.get_header_value("Content-Type").rfind("application/json", 0) == 0) {
        json body = json::parse(request.body);
        if (!body.contains("input_url") || body["input_url"].empty())
          throw std::runtime_error("JSON submissions require input_url");
        status["input_url"] = body["input_url"];
      } else if (!request.body.empty()) {
        // A job tar: validated and unpacked now, so malformed submissions
        // fail the request instead of the job.
        status["spec"] = extract_job_tar(request.body, state.job_dir(id) / "input");
      } else {
        throw std::runtime_error("empty body: send a job tar or JSON with input_url");
      }
    } catch (const std::exception &error) {
      return send_json(response, 400, {{"error", error.what()}});
    }

    state.write_status(id, status);
    {
      std::lock_guard guard(state.lock);
      state.queue.push_back(id);
    }
    state.wake.notify_one();
    send_json(response, 202, {{"id", id}, {"state", "queued"}});
  });

  server.set_logger([](const httplib::Request &request, const httplib::Response &response) {
    std::fprintf(stderr, "%s \"%s %s\" %d\n", request.remote_addr.c_str(), request.method.c_str(),
                 request.path.c_str(), response.status);
  });

  signal(SIGPIPE, SIG_IGN);  // dropped client connections must not kill the server
  json gpu = query_gpu();
  std::fprintf(stderr, "trtc builder on %s:%d — gpu=%s cc=%s driver=%s data=%s\n", host.c_str(), port,
               gpu["gpu_name"].is_null() ? "null" : std::string(gpu["gpu_name"]).c_str(),
               gpu["compute_capability"].is_null() ? "null" : std::string(gpu["compute_capability"]).c_str(),
               gpu["driver_version"].is_null() ? "null" : std::string(gpu["driver_version"]).c_str(),
               data_dir.c_str());
  if (!server.listen(host, port)) {
    std::fprintf(stderr, "cannot listen on %s:%d\n", host.c_str(), port);
    return 1;
  }
  return 0;
}
