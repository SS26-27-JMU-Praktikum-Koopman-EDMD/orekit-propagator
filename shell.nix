{ pkgs ? import <nixpkgs> {} }:

let
  python = pkgs.python313.withPackages (ps: with ps; [
    pip
    virtualenv
  ]);
in
pkgs.mkShell {
  name = "Koopman EDMD of Grav Harmonics shell";
  buildInputs = [
    python
    pkgs.jdk
    pkgs.stdenv.cc.cc.lib
    pkgs.wget
  ];

  # This shell is kind of ugly ngl
  shellHook = ''
    export LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH

    if [ ! -d .venv ]; then
      echo "python .venv not present"
      python -m venv .venv
      source ./.venv/bin/activate
      pip install -r requirements.txt
      pip install jupyter ipykernel
      python -m ipykernel install --prefix=.venv --name=koopman --display-name "Python (koopman)"
    else
      echo "python .venv present"
      source ./.venv/bin/activate
    fi
    if [ ! -f data/orekit-data.zip ]; then
       echo "data/orekit-data-main.zip not found"
       wget -O data/orekit-data.zip https://gitlab.orekit.org/orekit/orekit-data/-/archive/main/orekit-data-main.zip
    else
       echo "src/orekit-data-main.zip present"
    fi
  '';
}
