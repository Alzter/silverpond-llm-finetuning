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
                          trl = final_.callPackage ./python-modules/trl/default.nix { };
                        };
                    };
                }
            )

        ];
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
#             scipy
#             einops
#             evaluate
#             rouge_score
        ]
        ))
    ];
}
