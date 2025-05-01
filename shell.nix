let
    pkgs = import <nixpkgs> {
        config = {
            enableCuda = true;
            allowUnfree = true;
        };

        overlays = [ # Modify Python library to have overrides
            (
                final: prev: rec {
                    python312 = prev.python312.override {
                        self = python312;
                        packageOverrides = final_: prev_: {
                          torch = final_.torch-bin.overrideAttrs(torch-binFinalAttrs: torch-binPrevAttrs: {
                            passthru = torch-binPrevAttrs.passthru // {
                              cudaPackages = pkgs.cudaPackages;
                              cudaSupport = true;
                            };
                          });
                          torchvision = final_.torchvision-bin;
                          torchaudio = final_.torchaudio-bin;
                          trl = final_.callPackage ./build/trl/default.nix { };
#                           unsloth = final_.callPackage ./build/unsloth/default.nix { };
#                           unsloth-zoo = final_.callPackage ./build/unsloth-zoo/default.nix { };
#                           tyro = final_.callPackage ./build/tyro/default.nix { };
#                           cut-cross-entropy = final_.callPackage ./build/cut-cross-entropy/default.nix { };
                        };
                    };
                }
            )

        ];
};
in
pkgs.mkShell {
    buildInputs = with pkgs; [
        tmux
        (python312.withPackages (p: with p; [
            ipykernel
            jupyter
            pip
            numpy
            pandas
            torch
            torchvision
            torchaudio
            tqdm
            matplotlib
            bitsandbytes
            transformers
            peft
            accelerate
            datasets
            trl
            scikit-learn
            pytest
            openpyxl
            xlrd
            imbalanced-learn
#             tyro
#             cut-cross-entropy
#             unsloth
#             unsloth-zoo
#             scipy
#             einops
#             evaluate
#             rouge_score
        ]
        ))
    ];
}
