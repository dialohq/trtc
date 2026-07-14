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
        # libs stay behind.
        tensorrtFor = pin:
          pkgs.stdenv.mkDerivation {
            pname = "tensorrt";
            inherit (pin) version;
            src = pkgs.fetchurl {inherit (pin) url sha256;};
            nativeBuildInputs = [pkgs.zstd]; # 11.1 ships .tar.zst; tar autodetects
            sourceRoot = "TensorRT-${pin.version}";
            dontConfigure = true;
            dontBuild = true;
            installPhase = ''
              mkdir -p $out/lib
              cp -a include $out/include
              cp -a lib/*.so* $out/lib/
            '';
            dontFixup = true; # manylinux binaries; leave their rpaths alone
          };

        # The GPU-container contract. libcuda always comes from the HOST
        # driver, injected by the container runtime. Nix binaries never read
        # the container's /etc/ld.so.cache, so the injection locations must be
        # on LD_LIBRARY_PATH: /usr/local/nvidia is the legacy mount,
        # /usr/lib/x86_64-linux-gnu is where nvidia-container-toolkit actually
        # places driver libs.
        driverLibraryPath = "/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/lib/x86_64-linux-gnu";

        # The HTTP broker alone — no TensorRT, cheap to build, used by CI for
        # the end-to-end test.
        trtc-broker = pkgs.stdenv.mkDerivation {
          pname = "trtc-broker";
          version = "0.1.0";
          src = ./trtc-server;
          nativeBuildInputs = [pkgs.cmake];
          buildInputs = [pkgs.openssl pkgs.nlohmann_json pkgs.httplib];
          cmakeFlags = ["-DTRTC_WITH_TENSORRT=OFF"];
        };

        # Both binaries, linked against one pinned TensorRT.
        serverFor = pin: let
          tensorrt = tensorrtFor pin;
        in
          pkgs.stdenv.mkDerivation {
            pname = "trtc-server";
            version = "0.1.0-trt${pin.version}";
            src = ./trtc-server;
            nativeBuildInputs = [pkgs.cmake];
            buildInputs = [pkgs.openssl pkgs.nlohmann_json pkgs.httplib cudart];
            cmakeFlags = [
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

        # The builder image: two static-feeling binaries and one TensorRT, and
        # nothing else. TRT sits in its own layer so image pushes reuse it.
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
                "SSL_CERT_FILE=${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt"
                # TensorRT dlopens pieces of itself (libnvinfer_builder_resource),
                # libcudart (shipped, nix store), and libcuda (host driver,
                # injected by the container runtime) — all resolved here.
                "LD_LIBRARY_PATH=${tensorrtFor pin}/lib:${pkgs.lib.getLib cudart}/lib:${driverLibraryPath}"
              ];
              Cmd = ["/bin/trtc-server" "serve" "--host" "0.0.0.0" "--port" "8080"];
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

            # Pushes every pinned builder image, tagged trt<major.minor>.
            # The version list lives here in the flake, nowhere else:
            #   nix run .#push-builder-images -- "user:token" ghcr.io/dialohq/trtc-builder
            push-builder-images = pkgs.writeShellApplication {
              name = "push-builder-images";
              text = ''
                creds=$1
                registry=$2
                ${pkgs.lib.concatStringsSep "\n" (pkgs.lib.mapAttrsToList (v: pin: ''
                  ${(imageFor pin).copyTo}/bin/copy-to --dest-creds "$creds" "docker://$registry:trt${v}"
                  echo "pushed $registry:trt${v}"
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
