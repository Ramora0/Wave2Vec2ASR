from datasets import load_dataset, DatasetDict, Audio, Dataset
from functools import partial

def to_dataset(ds):
    def gen_from_iterable_dataset(ds):
        yield from ds

    return Dataset.from_generator(partial(gen_from_iterable_dataset, ds), 
                                  features=ds.features)

common_voice = DatasetDict()

common_voice["train"] = to_dataset(load_dataset("mozilla-foundation/common_voice_17_0","en",
                                                split="train", streaming=True, 
    token="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")
    .shuffle(seed=5318008)
    .take(64000))

common_voice["test"] = to_dataset(load_dataset("mozilla-foundation/common_voice_17_0","en",
                                               split="test", streaming=True, 
    token="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")
    .shuffle(seed=5318008)
    .take(1600))

common_voice["validation"] = to_dataset(load_dataset("mozilla-foundation/common_voice_17_0","en",
                                               split="validation", streaming=True, 
    token="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")
    .shuffle(seed=5318008)
    .take(3200))

common_voice = common_voice.remove_columns(["accent", "age", "client_id", "down_votes", 
                                            "gender", "locale", "path", "segment", 
                                            "up_votes", "variant"])
common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))

common_voice.save_to_disk("data/common_voice")