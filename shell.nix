{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
    name="impurePythonEnv";
    venvDir="./.venv";
    packages = [

        # Python Dependencies
        (pkgs.python312.withPackages(pypkgs: with pypkgs; [

            pip
            numpy
            pandas
            torch
            torchvision
            torchaudio
            ipykernel
            jupyter
            datasets

        ]))

    ];

    # SOURCE: https://www.reddit.com/r/NixOS/comments/1706v6p/comment/k3jry2z/
    # Run this command, only after creating the virtual environment
    postVenvCreation = ''
        unset SOURCE_DATE_EPOCH

        python -m ipykernel install --user --name=myenv4 --display-name="myenv4"
        pip install -r requirements.txt
    '';

    # Now we can execute any commands within the virtual environment.
    # This is optional and can be left out to run pip manually.
    postShellHook = ''
        # allow pip to install wheels
        unset SOURCE_DATE_EPOCH
    '';

}
