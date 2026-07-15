{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    nix2container.url = "github:nlewo/nix2container";
    nix2container.inputs.nixpkgs.follows = "nixpkgs";
    vast-cli.url = "github:dialohq/vast-cli.nix";
    vast-cli.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = {
    nixpkgs,
    flake-utils,
    nix2container,
    vast-cli,
    ...
  }:
    flake-utils.lib.eachDefaultSystem (
      system: let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true; # CUDA runtime (redistributable) for the TensorRT headers/libs
        };
        n2c = nix2container.packages.${system}.nix2container;
        cudart = pkgs.cudaPackages.cuda_cudart;
        # Just libcudart, without the nixpkgs cuda package's setup-hook and
        # header baggage (patchelf, cccl, ...) leaking into the image closure.
        cudartLibs = pkgs.runCommand "cudart-libs" {} ''
          mkdir -p $out/lib
          cp -a ${pkgs.lib.getLib cudart}/lib/libcudart.so* $out/lib/
        '';

        # The trtc release version: reported by /info and part of the image
        # tags (1.0.0-trt10.13).
        trtcVersion = "1.0.0";

        # The supported TensorRT versions: NVIDIA's official tarballs (lib/ +
        # include/ in one artifact), pinned by hash. Engines are
        # TRT-version-locked, so CI builds one image per entry, tagged
        # trt<major.minor>. From 10.16 on the builder resource is sharded per
        # GPU architecture; each arch listed here additionally gets a slim
        # single-arch image, tagged trt<major.minor>-sm<arch> — it builds on
        # that architecture family only.
        tensorrtPins = {
          "10.13" = {
            version = "10.13.3.9";
            url = "https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.13.3/tars/TensorRT-10.13.3.9.Linux.x86_64-gnu.cuda-12.9.tar.gz";
            sha256 = "a40650fd51f096969db072edf216a5026e61c71157bc87f475a8549681e09d34";
            archs = []; # monolithic builder resource; no per-arch variants
          };
          "10.16" = {
            version = "10.16.1.11";
            url = "https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.16.1/tars/TensorRT-10.16.1.11.Linux.x86_64-gnu.cuda-12.9.tar.gz";
            sha256 = "48e64fc9231fe3aeaa18be13385b486ea7c7131dd64f09b54e5fcb051386f014";
            archs = ["80" "86" "89" "90" "100" "120"];
          };
          "11.1" = {
            version = "11.1.0.106";
            url = "https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/11.1.0/tars/TensorRT-Enterprise-11.1.0.106-Linux-x86_64-cuda-12.9-Release-external.tar.zst";
            sha256 = "a38e9c87cb2f66b30ee03cf4ece3fc2ba94ae306afe675713991af0d30914530";
            archs = ["80" "86" "89" "90" "100" "120"];
          };
        };

        # Shared libraries + headers out of the tarball; the multi-GB static
        # libs stay behind. TensorRT dlopens pieces of itself
        # (libnvinfer_builder_resource), libcudart, and libcuda (host driver)
        # by bare soname — bake all of that into libnvinfer's RUNPATH so no
        # LD_LIBRARY_PATH is ever needed, in the image or out of it.
        tensorrtFor = pin:
          pkgs.stdenv.mkDerivation {
            pname = "tensorrt";
            inherit (pin) version;
            src = pkgs.fetchurl {inherit (pin) url sha256;};
            nativeBuildInputs = [pkgs.zstd pkgs.patchelf]; # 11.1 ships .tar.zst; tar autodetects
            sourceRoot = "TensorRT-${pin.version}";
            outputs = ["out" "dev"]; # headers compile the binaries, never ship
            dontConfigure = true;
            dontBuild = true;
            installPhase = ''
              mkdir -p $out/lib $dev
              cp -a include $dev/include
              # The *_win* builder resources (monolithic in TRT 10.13, one
              # per SM arch from 10.16 on — gigabytes of them) only serve
              # runtime_platform WINDOWS_AMD64 cross-builds; asking for that
              # fails loudly.
              rm -f lib/*_win*
              cp -a lib/*.so* $out/lib/
              for lib in $out/lib/libnvinfer.so.* $out/lib/libnvonnxparser.so.*; do
                [ -L "$lib" ] || patchelf --set-rpath \
                  '$ORIGIN:${cudartLibs}/lib:${driverLibraryPath}' "$lib"
              done
            '';
            dontFixup = true; # manylinux binaries; only the targeted rpath above
          };

        # A single-arch TensorRT: the universal extraction, minus every
        # builder resource except this architecture's (no PTX fallback either
        # — the variant builds for sm<arch> and nothing else, and fails
        # loudly elsewhere). A plain copy: the multi-GB tarball is only ever
        # decompressed once per version.
        tensorrtArchFor = pin: arch:
          pkgs.runCommand "tensorrt-sm${arch}-${pin.version}" {} ''
            mkdir -p $out/lib
            cp -a ${tensorrtFor pin}/lib/. $out/lib/
            chmod u+w $out/lib # cp -a keeps the store's read-only dir mode
            for f in $out/lib/libnvinfer_builder_resource*; do
              case "$f" in *_sm${arch}.so*) ;; *) rm -f "$f" ;; esac
            done
            ls $out/lib/libnvinfer_builder_resource_sm${arch}.so* >/dev/null # arch must exist in this TensorRT
          '';

        trtFor = pin: arch:
          if arch == null
          then tensorrtFor pin
          else tensorrtArchFor pin arch;

        # The GPU-container contract. libcuda always comes from the HOST
        # driver, injected by the container runtime at these FHS locations
        # (/usr/local/nvidia is the legacy mount, /usr/lib/x86_64-linux-gnu is
        # where nvidia-container-toolkit actually places driver libs). Nix
        # binaries never read /etc/ld.so.cache, so these paths are baked into
        # TensorRT's RUNPATH and tried directly by our dlopen.
        driverLibraryPath = "/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/lib/x86_64-linux-gnu";

        # The HTTP broker alone — no TensorRT, cheap to build, used by CI for
        # the end-to-end test.
        trtc-broker = pkgs.stdenv.mkDerivation {
          pname = "trtc-broker";
          version = trtcVersion;
          src = ./trtc-server;
          nativeBuildInputs = [pkgs.cmake];
          buildInputs = [pkgs.openssl pkgs.nlohmann_json pkgs.httplib pkgs.libarchive];
          cmakeFlags = ["-DTRTC_WITH_TENSORRT=OFF" "-DTRTC_VERSION=${trtcVersion}"];
        };

        # trtc-server (self-contained: serving + its own build subcommand for
        # jobs) and the standalone trtc-build, linked against one pinned
        # TensorRT.
        serverFor = pin: arch: let
          tensorrt = trtFor pin arch;
        in
          pkgs.stdenv.mkDerivation {
            pname = "trtc-server${pkgs.lib.optionalString (arch != null) "-sm${arch}"}";
            version = "${trtcVersion}-trt${pin.version}";
            src = ./trtc-server;
            nativeBuildInputs = [pkgs.cmake];
            buildInputs = [pkgs.openssl pkgs.nlohmann_json pkgs.httplib pkgs.libarchive cudart];
            cmakeFlags = [
              "-DTRTC_VERSION=${trtcVersion}"
              "-DTENSORRT_INCLUDE_DIR=${(tensorrtFor pin).dev}/include"
              "-DTENSORRT_NVINFER=${tensorrt}/lib/libnvinfer.so"
              "-DTENSORRT_NVONNXPARSER=${tensorrt}/lib/libnvonnxparser.so"
              # trtc-build finds its TensorRT in the store forever, no
              # LD_LIBRARY_PATH involved.
              "-DCMAKE_INSTALL_RPATH=${tensorrt}/lib"
            ];
            # The crt/ headers cuda_runtime_api.h pulls in ship with nvcc;
            # only the headers are used, no nvcc anywhere.
            env.NIX_CFLAGS_COMPILE = "-isystem ${pkgs.lib.getDev pkgs.cudaPackages.cuda_nvcc}/include";
          };

        # The bare-image survival kit for stock runtimes: glibc needs
        # nsswitch.conf to resolve DNS (presigned S3 URLs), some tooling wants
        # a passwd entry, and /tmp + /data must exist even before a volume is
        # mounted. No shell, no coreutils — this is the whole filesystem.
        fsRoot = pkgs.runCommand "trtc-fs" {} ''
          mkdir -p $out/etc $out/tmp $out/data $out/root
          echo "hosts: files dns" > $out/etc/nsswitch.conf
          echo "root:x:0:0:root:/root:/bin/trtc-server" > $out/etc/passwd
          echo "root:x:0:" > $out/etc/group
        '';

        # The builder image: one binary and one TensorRT, and nothing else.
        # TRT sits in its own layer so image pushes reuse it.
        imageFor = pin: arch:
          n2c.buildImage {
            name = "trtc-builder";
            copyToRoot = [
              (pkgs.buildEnv {
                name = "trtc-bin";
                paths = [(serverFor pin arch)];
                pathsToLink = ["/bin"];
              })
              fsRoot
            ];
            perms = [
              {
                path = fsRoot;
                regex = "/tmp";
                mode = "1777";
              }
              {
                path = fsRoot;
                regex = "/data";
                mode = "0755";
              }
            ];
            layers = [(n2c.buildLayer {deps = [(trtFor pin arch) cudartLibs];})];
            config = {
              Env = [
                "PATH=/bin"
                "USER=root"
                "HOME=/root"
                "NVIDIA_VISIBLE_DEVICES=all"
                "NVIDIA_DRIVER_CAPABILITIES=compute,utility"
                "TRTC_DATA_DIR=/data"
                "TRTC_TENSORRT_VERSION=${pin.version}"
                "SSL_CERT_FILE=${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt"
                # No LD_LIBRARY_PATH: every library location is baked into the
                # binaries' and TensorRT's RUNPATHs.
              ];
              Cmd = ["/bin/trtc-server"];
              ExposedPorts."8080/tcp" = {};
              Volumes."/data" = {};
            };
          };

        # One-shot local build without composing a spec directory first:
        #   nix run github:dialohq/trtc#build-10.13 -- spec.json model.onnx [data files...] [--out DIR]
        # Files are staged (symlinked) into a scratch dir under the canonical
        # names and handed to the pinned `trtc-server build`.
        buildAppFor = pin:
          pkgs.writeShellApplication {
            name = "trtc-build-once";
            runtimeInputs = [pkgs.coreutils];
            text = ''
              if [ "$#" -lt 2 ]; then
                echo "usage: nix run .#build-<trt> -- <spec.json> <model.onnx> [onnx data files...] [--out DIR]" >&2
                exit 2
              fi
              out="$PWD"
              spec=""
              files=()
              while [ "$#" -gt 0 ]; do
                case "$1" in
                  --out) out="$2"; shift 2 ;;
                  *) if [ -z "$spec" ]; then spec="$1"; else files+=("$1"); fi; shift ;;
                esac
              done
              work="$(mktemp -d)"
              trap 'rm -rf "$work"' EXIT
              cp "$spec" "$work/trtc_build_spec.json"
              for f in "''${files[@]}"; do ln -s "$(realpath "$f")" "$work/$(basename "$f")"; done
              exec ${serverFor pin null}/bin/trtc-build "$work" --out "$out"
            '';
          };

        # "10.13" -> "trtc-builder-trt10-13" (flake attr names can't have dots)
        attrName = prefix: v: "${prefix}${builtins.replaceStrings ["."] ["-"] v}";
      in {
        packages =
          rec {
            default = trtc-broker;
            inherit trtc-broker;

            # Prebuilt env for `trtc launch` — trtc resolved once at build time,
            # vastai from the vast-cli input (no runtime installs).
            launch-env = pkgs.stdenv.mkDerivation {
              name = "trtc-launch-env";
              src = ./trtc;
              __noChroot = true;
              dontFixup = true;
              NIX_SSL_CERT_FILE = "${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt";
              SSL_CERT_FILE = "${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt";
              nativeBuildInputs = [pkgs.python311 pkgs.uv];
              buildPhase = ''
                runHook preBuild
                export HOME="$TMPDIR" UV_CACHE_DIR="$TMPDIR/uv"
                export UV_PYTHON_PREFERENCE=only-system UV_PYTHON=${pkgs.python311}/bin/python3.11
                cp -r "$src" ./trtc-src && chmod -R u+w ./trtc-src
                uv venv "$out"
                uv pip install --python "$out/bin/python" ./trtc-src
                runHook postBuild
              '';
              dontInstall = true;
            };

            # Rents a vast.ai GPU and starts the builder image matching the
            # caller's TensorRT; prints `export TRTC_BUILDER=...`:
            #   eval "$(nix run github:dialohq/trtc#launch-builder)"
            launch-builder = pkgs.writeShellApplication {
              name = "launch-builder";
              runtimeInputs = [vast-cli.packages.${system}.default];
              text = ''exec ${launch-env}/bin/trtc launch "$@"'';
            };

            # Pushes every pinned builder image — universal and single-arch —
            # each under three tags:
            #   trt10.13[-sm120]                          moving latest
            #   1.0.0-trt10.13[-sm120]                    this trtc release
            #   1.0.0-trt10.13[-sm120]-<nix image hash>   immutable
            # The version and arch lists live here in the flake, nowhere else:
            #   nix run .#push-builder-images -- "user:token" ghcr.io/dialohq/trtc-builder
            push-builder-images = pkgs.writeShellApplication {
              name = "push-builder-images";
              text = ''
                # Host registries.conf may be v1 (GitHub runners ship one),
                # which skopeo rejects; carry our own empty (= defaults) v2.
                export CONTAINERS_REGISTRIES_CONF=${pkgs.writeText "registries.conf" ""}
                creds=$1
                registry=$2
                ${pkgs.lib.concatStringsSep "\n" (pkgs.lib.flatten (pkgs.lib.mapAttrsToList (
                  v: pin:
                    map (arch: let
                      image = imageFor pin arch;
                      base = "trt${v}${pkgs.lib.optionalString (arch != null) "-sm${arch}"}";
                    in ''
                      for tag in "${base}" "${trtcVersion}-${base}" "${trtcVersion}-${base}-${image.imageTag}"; do
                        ${image.copyTo}/bin/copy-to --dest-creds "$creds" "docker://$registry:$tag"
                        echo "pushed $registry:$tag"
                      done
                    '') ([null] ++ pin.archs)
                )
                tensorrtPins))}
              '';
            };
          }
          # trtc-server-trt10-13 (binaries) and trtc-builder-trt10-13 (image)
          # per supported TensorRT version.
          // pkgs.lib.mapAttrs' (v: pin: pkgs.lib.nameValuePair (attrName "trtc-server-trt" v) (serverFor pin null))
          tensorrtPins
          // pkgs.lib.mapAttrs' (v: pin: pkgs.lib.nameValuePair (attrName "trtc-builder-trt" v) (imageFor pin null))
          tensorrtPins
          // pkgs.lib.mapAttrs' (v: pin: pkgs.lib.nameValuePair "build-${v}" (buildAppFor pin))
          tensorrtPins
          # Single-arch images: trtc-builder-trt11-1-sm120 etc.
          // builtins.listToAttrs (pkgs.lib.flatten (pkgs.lib.mapAttrsToList (
            v: pin:
              map (arch: pkgs.lib.nameValuePair (attrName "trtc-builder-trt" v + "-sm${arch}") (imageFor pin arch))
              pin.archs
          )
          tensorrtPins));

        # `nix run .#build-10.13` unquoted: nix splits attrpaths on dots, so
        # mirror the build apps as nested attrs (build-10.13 -> build-10 . 13).
        legacyPackages = builtins.foldl' pkgs.lib.recursiveUpdate {} (pkgs.lib.mapAttrsToList (
            v: pin: let
              parts = pkgs.lib.splitString "." v;
            in {"build-${builtins.elemAt parts 0}"."${builtins.elemAt parts 1}" = buildAppFor pin;}
          )
          tensorrtPins);
      }
    );
}
