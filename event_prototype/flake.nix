{
  description = "elara3 event queue prototype: async event queue, LLM triage, subagent dispatch";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        python = pkgs.python312;
      in
      {
        devShells.default = pkgs.mkShell {
          packages = [ python pkgs.uv pkgs.sqlite ];

          env = {
            # Resolve against the interpreter from this shell rather than a
            # downloaded one, so the venv matches what Nix provides.
            UV_PYTHON = python.interpreter;
            UV_PYTHON_DOWNLOADS = "never";
            # Wheels with compiled extensions link against the host libc++.
            LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc.lib pkgs.zlib ];
          };

          shellHook = ''
            echo "event_prototype devShell — run 'uv sync' then 'uv run python -m event_prototype --help'"
          '';
        };
      });
}
