let
    pkgs = import <nixpkgs> {
        config = {
            enableCuda = true;
            allowUnfree = true;
        };

#          overlays = [ # Modify Python library to have overrides
#            (
#              final: prev: rec {
#                python312 = prev.python312.override {
#                  self = python312;
#                  packageOverrides = final_: prev_: {
#                    accelerate = prev_.accelerate.override {
#                      torch = final_.torch-bin; # Change the torch library referenced in transformers to torch-bin (precompiled)
#                    };
#                  };
#                };
#              }
#            )
#          ];
};
in
pkgs.mkShell {
    buildInputs = with pkgs; [
        (python312.withPackages (p: with p; [
            ipykernel
            jupyter
            pip
            numpy
            pandas
#            torch-bin
#            torchvision-bin
#            torchaudio-bin
            tqdm
            matplotlib
            bitsandbytes
            transformers
            peft
            accelerate
            datasets
#             trl
#             scipy
#             einops
#             evaluate
#             rouge_score
        ]
        ))
    ];
}
