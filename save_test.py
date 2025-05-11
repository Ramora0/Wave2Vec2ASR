import torch
from transformers import Wav2Vec2ForCTC
from MagnetWav2Vec2 import MagnetWav2Vec2

print("Creating model...")
model = Wav2Vec2ForCTC.from_pretrained(
    "./data/models/wav2vec2-librispeech-1h",
    ctc_loss_reduction="mean", 
    pad_token_id=28,
    token="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"
)
model.freeze_feature_extractor()

model.__class__ = MagnetWav2Vec2
model.load_magnet(1, .67, 0.5)

mlp_state_dict_before = model.wav2vec2.boundary_predictor.boundary_mlp.state_dict()

print("Saving model...")
model.save_pretrained("./data/models/magnet-test")
print("Loading model...")
model = MagnetWav2Vec2.from_pretrained("./data/models/magnet-test")

mlp_state_dict_after = model.wav2vec2.boundary_predictor.boundary_mlp.state_dict()

print("Comparing model...")
are_equal = True
if mlp_state_dict_before.keys() != mlp_state_dict_after.keys():
    are_equal = False
    print("\nMLP state dict keys differ!")
else:
    for key in mlp_state_dict_before:
        if not torch.equal(mlp_state_dict_before[key], mlp_state_dict_after[key]):
            are_equal = False
            print(f"\nDifference found in parameter: {key}")
            print("Before:", mlp_state_dict_before[key])
            print("After:", mlp_state_dict_after[key])
        else:
            print(f"Parameter {key} is the same.")

if are_equal:
    print("\nComparison Result: Both MLP layers are the same.")
else:
    print("\nComparison Result: The MLP layers are different.")