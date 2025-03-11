let
    pkgs = import <nixpkgs> {
        config = {
            enableCuda = false;
            allowUnfree = true;
        };

#         overlays = [ # Modify Python library to have overrides
#
#             (
#                 final: prev: rec {
#                 python312 = prev.python312.override {
#                     self = python312;
#                     packageOverrides = final_: prev_: {
#                     transformers = prev_.transformers.override {
#                         torch = final_.torch-bin; # Change the torch library referenced in transformers to torch-bin (precompiled)
#                     };
#                     };
#                 };
#             }
#             )
#
#         ];
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
            torch-bin
            torchvision-bin
            torchaudio-bin
            tqdm
#             bitsandbytes
            transformers
#             peft
#             accelerate
            datasets
#             scipy
#             einops
#             evaluate
#             trl
#             rouge_score
        ]
        ))
    ];
}
