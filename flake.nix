{
  description = "Homotopy Neural Networks: Agda + Haskell + Python development";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
    haskell-flake.url = "github:srid/haskell-flake";
    agda = {
      url = "github:agda/agda/5cb475a67135ca4ce42428c6f0294cea58a3ca2b";
      flake = false;
    };
    onelab = {
      url = "github:the1lab/1lab/e51776c97deb6faffa48b8d74e1542e43f1d8a";
      flake = false;
    };
  };

  outputs = inputs@{ self, nixpkgs, flake-parts, haskell-flake, agda, onelab }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = nixpkgs.lib.systems.flakeExposed;
      imports = [ haskell-flake.flakeModule ];

      perSystem = { self', pkgs, system, ... }:
      let
        # Python with ML/AI stack
        pythonEnv = pkgs.python312.withPackages (ps: with ps; [
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
        ]);

        # Agda with 1Lab and custom GitHub version
        agdaWithPackages = (pkgs.agdaPackages.override {
          Agda = pkgs.haskellPackages.Agda;
        }).agda.withPackages (p: [
          (p._1lab.overrideAttrs (oldAttrs: {
            version = "github-latest";
            src = onelab;
          }))
        ]);

      in
      {
        # Haskell project for einsum-bridge
        haskellProjects.default = {
          basePackages = pkgs.haskell.packages.ghc910;

          # hask/ directory will be auto-discovered as local package
          packages = {
            # Override Agda package to use our custom version
            Agda.source = agda;
          };

          settings = {
            einsum-bridge = {
              # Package-specific settings if needed
            };
          };

          devShell = {
            tools = hp: {
              hlint = null; # Disable hlint due to GHC 9.10 incompatibility
              haskell-language-server = null; # Disable HLS
            };
          };
        };

        # Override devShell to include Agda + Python
        devShells.default = pkgs.mkShell {
          inputsFrom = [
            self'.devShells.default  # Haskell tools from haskell-flake
          ];

          packages = [
            agdaWithPackages
            pythonEnv
          ];

          shellHook = ''
            export AGDA_DIR="$(pwd)"
            echo "Homotopy Neural Networks Dev Environment"
            echo "========================================"
            echo "Agda: $(agda --version)"
            echo "GHC: $(ghc --version)"
            echo "Cabal: $(cabal --version | head -1)"
            echo "Python: $(python3 --version)"
            echo ""
            echo "Available commands:"
            echo "  cabal run einsum-repl  - Interactive Haskell/Python bridge"
            echo "  python3 python-runtime/test_session.py  - Test Python backend"
            echo "  agda --library-file=./libraries <file>   - Type-check Agda"
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
      };

      # Apply Haskell package overlay for Agda from GitHub
      flake.overlays.default = final: prev: {
        haskellPackages = prev.haskellPackages.override {
          overrides = hself: hsuper: {
            Agda = (hself.callCabal2nix "Agda" agda {}).overrideAttrs (oldAttrs: oldAttrs // {
              meta = oldAttrs.meta // {
                mainProgram = "agda";
              };
            });
          };
        };
      };
    };
}
