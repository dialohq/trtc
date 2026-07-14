// One binary. `trtc-server` (or `trtc-server serve`) runs the HTTP broker;
// `trtc-server build <target>` builds engines — the broker execs its own
// binary for jobs, so the image and a local `nix build` result are the same
// self-contained thing.

#include <cstdio>
#include <string>

int serve_main(int argc, char **argv);
#ifdef TRTC_WITH_TENSORRT
int build_main(int argc, char **argv);
#endif

int main(int argc, char **argv) {
  if (argc > 1 && std::string(argv[1]) == "build") {
#ifdef TRTC_WITH_TENSORRT
    return build_main(argc - 1, argv + 1);
#else
    std::fprintf(stderr, "this trtc-server was built without TensorRT (broker only)\n");
    return 2;
#endif
  }
  return serve_main(argc, argv);  // serving is the default; serve_main tolerates a literal "serve"
}
