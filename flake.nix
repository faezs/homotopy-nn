{
  description = "Agda development environment with Cubical Agda";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in
      {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            (agda.withPackages (p: [
              p._1lab
            ]))
          ];

          shellHook = ''
            export AGDA_DIR="$(pwd)"
            echo "Agda development environment loaded!"
            echo "Available packages:"
            echo "  - Agda: $(agda --version)"
            echo "  - Cubical Agda library"
            echo "  - Agda standard library"
            echo "  - 1lab library"
            echo "AGDA_DIR set to: $AGDA_DIR"
          '';
        };
      });
}
