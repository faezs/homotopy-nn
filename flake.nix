{
  description = "Agda development environment with Cubical Agda";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    agda = {
      url = "github:agda/agda/5cb475a67135ca4ce42428c6f0294cea58a3ca2b";
      flake = false;
    };
    onelab = {
      url = "github:the1lab/1lab/e51776c97deb6faffa48b8d74e1542e43f1d8a";
      flake = false;
    };
  };

  outputs = { self, nixpkgs, flake-utils, agda, onelab }:
    
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [
            (final: prev: {
              haskellPackages = prev.haskellPackages.override {
                overrides = hself: hsuper: {
                  # Build Agda from GitHub source
                  Agda = (hself.callCabal2nix "Agda" agda {}).overrideAttrs (oldAttrs: oldAttrs // {
                    meta = oldAttrs.meta // {
                      mainProgram = "agda";
                    };
                  });
                };
              };
            })
          ];
        };
      in
      {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            ((agdaPackages.override {
              Agda = haskellPackages.Agda;  # Use our GitHub Agda
            }).agda.withPackages (p: [
              (p._1lab.overrideAttrs (oldAttrs: {
                version = "github-latest";
                src = onelab;
              }))
            ]))
            (python312.withPackages (ps: with ps; [
              numpy
              scipy
              networkx
              scikit-learn
              pyyaml
              gudhi
              torch
              transformers
              tokenizers
              datasets
              safetensors
              accelerate
              huggingface-hub
              onnx
              onnxruntime
              jax
              jaxlib
              flax
              optax
              matplotlib
              tqdm
              pytest
              hypothesis
              tensorboard
            ]))
          ];

          shellHook = ''
            export AGDA_DIR="$(pwd)"
            echo "Agda + Python dev environment loaded!"
            echo "Agda: $(agda --version)"
            echo "Python: $(python3 --version)"
            echo "Python site: $(python3 -c 'import sys; print(sys.executable)')"
            echo "1Lab available; Python has numpy/scipy/networkx/sklearn + HF stack"
            echo ""
            echo "Run topos tests: cd neural_compiler/topos && ./run_topos_tests.sh"
          '';
        };

        # Test suite for topos categorical structure
        checks.topos-tests = pkgs.stdenv.mkDerivation {
          name = "topos-property-tests";
          src = ./neural_compiler/topos;
          buildInputs = [
            (pkgs.python312.withPackages (ps: with ps; [
              pytest
              hypothesis
              torch
              numpy
            ]))
          ];
          buildPhase = ''
            export PYTHONPATH="${./neural_compiler}:$PYTHONPATH"
            python3 -m pytest test_topos_laws.py -v --tb=short
          '';
          installPhase = ''
            mkdir -p $out
            echo "Topos tests passed" > $out/result
          '';
        };

        # Run all tests
        checks.all-tests = pkgs.stdenv.mkDerivation {
          name = "all-topos-tests";
          src = ./neural_compiler/topos;
          buildInputs = [
            (pkgs.python312.withPackages (ps: with ps; [
              pytest
              hypothesis
              torch
              numpy
            ]))
          ];
          buildPhase = ''
            export PYTHONPATH="${./neural_compiler}:$PYTHONPATH"
            python3 -m pytest test_topos_laws.py test_topos_properties.py -v --hypothesis-show-statistics --tb=short
          '';
          installPhase = ''
            mkdir -p $out
            echo "All topos tests passed" > $out/result
          '';
        };
      });
}
