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

        # The trtc release version: reported by /info and part of the image
        # tags (1.0.0-trt10.13).
        trtcVersion = "1.0.0";

        # The supported TensorRT versions: NVIDIA's official tarballs (lib/ +
        # include/ in one artifact), pinned by hash. Engines are
        # TRT-version-locked, so CI builds one image per entry, tagged
        # trt<major.minor>.
        tensorrtPins = {
          "10.13" = {
            version = "10.13.3.9";
            url = "https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.13.3/tars/TensorRT-10.13.3.9.Linux.x86_64-gnu.cuda-12.9.tar.gz";
            sha256 = "a40650fd51f096969db072edf216a5026e61c71157bc87f475a8549681e09d34";
          };
          "10.16" = {
            version = "10.16.1.11";
            url = "https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.16.1/tars/TensorRT-10.16.1.11.Linux.x86_64-gnu.cuda-12.9.tar.gz";
            sha256 = "48e64fc9231fe3aeaa18be13385b486ea7c7131dd64f09b54e5fcb051386f014";
          };
          "11.1" = {
            version = "11.1.0.106";
            url = "https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/11.1.0/tars/TensorRT-Enterprise-11.1.0.106-Linux-x86_64-cuda-12.9-Release-external.tar.zst";
            sha256 = "a38e9c87cb2f66b30ee03cf4ece3fc2ba94ae306afe675713991af0d30914530";
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
            dontConfigure = true;
            dontBuild = true;
            installPhase = ''
              mkdir -p $out/lib
              cp -a include $out/include
              cp -a lib/*.so* $out/lib/
              for lib in $out/lib/libnvinfer.so.* $out/lib/libnvonnxparser.so.*; do
                [ -L "$lib" ] || patchelf --set-rpath \
                  '$ORIGIN:${pkgs.lib.getLib cudart}/lib:${driverLibraryPath}' "$lib"
              done
            '';
            dontFixup = true; # manylinux binaries; only the targeted rpath above
          };

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

        # Both binaries, linked against one pinned TensorRT.
        serverFor = pin: let
          tensorrt = tensorrtFor pin;
        in
          pkgs.stdenv.mkDerivation {
            pname = "trtc-server";
            version = "${trtcVersion}-trt${pin.version}";
            src = ./trtc-server;
            nativeBuildInputs = [pkgs.cmake];
            buildInputs = [pkgs.openssl pkgs.nlohmann_json pkgs.httplib pkgs.libarchive cudart];
            cmakeFlags = [
              "-DTRTC_VERSION=${trtcVersion}"
              "-DTENSORRT_INCLUDE_DIR=${tensorrt}/include"
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
        imageFor = pin:
          n2c.buildImage {
            name = "trtc-builder";
            copyToRoot = [
              (pkgs.buildEnv {
                name = "trtc-bin";
                paths = [(serverFor pin)];
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
            layers = [(n2c.buildLayer {deps = [(tensorrtFor pin) (pkgs.lib.getLib cudart)];})];
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

            # Pushes every pinned builder image under three tags:
            #   trt10.13                          the moving latest for that TRT line
            #   1.0.0-trt10.13                    this trtc release
            #   1.0.0-trt10.13-<nix image hash>   immutable, content-addressed
            # The version list lives here in the flake, nowhere else:
            #   nix run .#push-builder-images -- "user:token" ghcr.io/dialohq/trtc-builder
            push-builder-images = pkgs.writeShellApplication {
              name = "push-builder-images";
              text = ''
                creds=$1
                registry=$2
                ${pkgs.lib.concatStringsSep "\n" (pkgs.lib.mapAttrsToList (v: pin: let
                  image = imageFor pin;
                in ''
                  for tag in "trt${v}" "${trtcVersion}-trt${v}" "${trtcVersion}-trt${v}-${image.imageTag}"; do
                    ${image.copyTo}/bin/copy-to --dest-creds "$creds" "docker://$registry:$tag"
                    echo "pushed $registry:$tag"
                  done
                '')
                tensorrtPins)}
              '';
            };
          }
          # trtc-server-trt10-13 (binaries) and trtc-builder-trt10-13 (image)
          # per supported TensorRT version.
          // pkgs.lib.mapAttrs' (v: pin: pkgs.lib.nameValuePair (attrName "trtc-server-trt" v) (serverFor pin))
          tensorrtPins
          // pkgs.lib.mapAttrs' (v: pin: pkgs.lib.nameValuePair (attrName "trtc-builder-trt" v) (imageFor pin))
          tensorrtPins;
      }
    );
}
