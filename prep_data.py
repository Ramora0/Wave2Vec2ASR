import aiohttp
from data_loader import get_dataset
from datasets import load_dataset, Audio, load_from_disk, concatenate_datasets
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor

scratch_path = "/fs/scratch/PAS2836/lees_stuff"

# dataset = load_dataset("openslr/librispeech_asr", trust_remote_code=True, storage_options={
#                        'client_kwargs': {'timeout': aiohttp.ClientTimeout(total=60*60*24)}})

# dataset = dataset.remove_columns(["file", "speaker_id", "chapter_id", "id"])
# dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

# combined_train = concatenate_datasets([
#     dataset["train.clean.100"],
#     dataset["train.clean.360"],
#     dataset["train.other.500"]
# ])
# dataset["train"] = combined_train

# del dataset["train.clean.100"]
# del dataset["train.clean.360"]
# del dataset["train.other.500"]
# del dataset["test.other"]
# del dataset["validation.other"]

# dataset.save_to_disk(f"{scratch_path}/librispeech-full")

dataset = load_from_disk(f"{scratch_path}/librispeech-full")

print(dataset["train"][0])

feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small",
                                                            token="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")

tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="English", task="transcribe",
                                             token="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")

processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="English", task="transcribe",
                                             token="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")

dataset = get_dataset(dataset, feature_extractor,
                      tokenizer, cache_dir="librispeech-full")
