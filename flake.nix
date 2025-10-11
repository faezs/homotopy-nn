{
  description = "Agda development environment with Cubical Agda";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
      in
      {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            (agda.withPackages (p: [ p._1lab ]))
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
            ]))
          ];

          shellHook = ''
            export AGDA_DIR="$(pwd)"
            echo "Agda + Python dev environment loaded!"
            echo "Agda: $(agda --version)"
            echo "Python: $(python3 --version)"
            echo "Python site: $(python3 -c 'import sys; print(sys.executable)')"
            echo "1Lab available; Python has numpy/scipy/networkx/sklearn + HF stack"
          '';
        };
      });
}
