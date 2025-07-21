import aiohttp
from datasets import load_dataset, DownloadConfig
from functools import partial

# Set DownloadConfig(timeout=300.0)

librispeech = load_dataset(
        "openslr/librispeech_asr",
        trust_remote_code=True,
        token="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",
    storage_options={'client_kwargs': {'timeout': aiohttp.ClientTimeout(total=60*60*24)}}
    )

print(librispeech)

# common_voice = common_voice.remove_columns(["accent", "age", "client_id", "down_votes", 
#                                             "gender", "locale", "path", "segment", 
#                                             "up_votes", "variant"])
# common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))

# common_voice.save_to_disk("data/common_voice")