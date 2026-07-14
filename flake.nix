{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    x2container.url = "github:dialohq/x2container.nix/filter-sync";
    x2container.inputs.nixpkgs.follows = "nixpkgs";
    nix2container.follows = "x2container/nix2container";
    vast-cli.url = "github:dialohq/vast-cli.nix";
    vast-cli.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = {
    nixpkgs,
    flake-utils,
    x2container,
    vast-cli,
    ...
  }:
    flake-utils.lib.eachDefaultSystem (
      system: let
        pkgs = import nixpkgs {inherit system;};
        gccLib = pkgs.stdenv.cc.cc.lib;
        # The GPU-container contract. libcuda/libnvidia-ml always come from
        # the HOST driver, injected by the container runtime; the image only
        # carries CUDA runtime libs. Nix binaries use the nix loader, which
        # never reads the container's /etc/ld.so.cache, so the injection
        # locations must be on LD_LIBRARY_PATH explicitly (nix libs first):
        # /usr/local/nvidia is the legacy mount, /usr/lib/x86_64-linux-gnu is
        # where nvidia-container-toolkit actually places driver libs.
        gpuContainer = {
          driverLibraryPath = ":/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/lib/x86_64-linux-gnu";
          env = [
            "NVIDIA_VISIBLE_DEVICES=all"
            "NVIDIA_DRIVER_CAPABILITIES=compute,utility"
          ];
        };
      in {
        packages = rec {
          default = trtc-builder;

          # The remote builder: a fixed, correct environment — trtc plus the
          # TensorRT the workspace lock pins (via the trtc-builder member's
          # trtc[builder] dependency). Pinned to one TensorRT version like a
          # derivation; a plan pinning a different version fails the job. CI
          # builds one image per supported version (see the workflow matrix),
          # tagged trt<major.minor>. Run:
          #   docker run --gpus all -p 8080:8080 -v trtc-data:/data \
          #       ghcr.io/dialohq/trtc-builder:trt<version>
          trtc-builder = x2container.lib.${system}.uv2container.buildImage {
            name = "trtc-builder";
            src = ./.;
            python = pkgs.python311;
            members = ["trtc" "trtc-builder"];
            # Only libstdc++ for the manylinux TRT libs; deliberately NOT nix
            # glibc — host-injected FHS binaries must not resolve a foreign
            # libc ahead of their own.
            runtimeLibs = [gccLib];
            extraLdLibraryPath = gpuContainer.driverLibraryPath;
            extraLibraryPath = gpuContainer.driverLibraryPath;
            config = {
              Env =
                gpuContainer.env
                ++ [
                  "USER=root"
                  "PYTHONUNBUFFERED=1"
                  "TRTC_DATA_DIR=/data"
                ];
              Cmd = ["python" "-m" "trtc.server.cli" "serve" "--host" "0.0.0.0" "--port" "8080"];
            };
          };

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
          # caller's TensorRT; prints `export TRTC_BUILDER=...`. Consumers run
          # it straight off this flake:
          #   eval "$(nix run github:dialohq/trtc#launch-builder -- --trt-version 11.1)"
          # Needs a vast.ai key (VAST_API_KEY or a configured vastai CLI).
          launch-builder = pkgs.writeShellApplication {
            name = "launch-builder";
            runtimeInputs = [vast-cli.packages.${system}.default];
            text = ''exec ${launch-env}/bin/trtc launch "$@"'';
          };
        };
      }
    );
}
