let
    pkgs = import <nixpkgs> {config = {
        enableCuda = false;
        allowUnfree = true;
    };};
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
#             bitsandbytes
            transformers
#             peft
#             accelerate
            datasets
#             scipy
#             einops
#             evaluate
            #trl
            #rouge_score
        ]
    ))
    ];
}
