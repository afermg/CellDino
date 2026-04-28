{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.11";
    systems.url = "github:nix-systems/default";
    flake-utils.url = "github:numtide/flake-utils";
    flake-utils.inputs.systems.follows = "systems";
    pynng-flake.url = "github:afermg/pynng";
    pynng-flake.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      systems,
      ...
    }@inputs:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          system = system;
          config = {
            allowUnfree = true;
            cudaSupport = true;
          };
        };
      in
      with pkgs;
      rec {
        apps.default =
          let
            # Build detectron2 without its (very slow) test phase: the check
            # phase pulls in timm/fairscale/etc. and can take >20 min under
            # nix without those being cached.
            detectron2-noCheck = python3.pkgs.detectron2.overrideAttrs (_: {
              doCheck = false;
              doInstallCheck = false;
            });
            python_with_pkgs = python3.withPackages (pp: [
              packages.nahual
              pp.torch
              pp.torchvision
              pp.numpy
              pp.pillow
              detectron2-noCheck
            ]);
            runServer = pkgs.writeScriptBin "runserver.sh" ''
              #!${pkgs.bash}/bin/bash
              ${python_with_pkgs}/bin/python ${self}/server.py ''${@:-"ipc:///tmp/celldino.ipc"}
            '';
          in
          {
            type = "app";
            program = "${runServer}/bin/runserver.sh";
          };

        packages = {
          nahual = pkgs.python3.pkgs.callPackage ./nix/nahual.nix {
            pynng = inputs.pynng-flake.packages.${system}.pynng;
          };
        };

        devShells = {
          default =
            let
              detectron2-noCheck = python3.pkgs.detectron2.overrideAttrs (_: {
                doCheck = false;
                doInstallCheck = false;
              });
              python_with_pkgs = python3.withPackages (pp: [
                packages.nahual
                pp.torch
                pp.torchvision
                pp.tifffile
                pp.scikit-image
                pp.scikit-learn
                pp.numpy
                pp.pyyaml
                detectron2-noCheck
              ]);
            in
            mkShell {
              packages = [
                python_with_pkgs
                pkgs.cudaPackages.cudatoolkit
              ];
              shellHook = ''
                export PYTHONPATH=${python_with_pkgs}/${python_with_pkgs.sitePackages}:$PYTHONPATH
              '';
            };
        };
      }
    );
}
