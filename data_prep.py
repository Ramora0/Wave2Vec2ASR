import random
from datasets import load_dataset

timit = load_dataset("timit_asr", trust_remote_code=True, data_dir="data")
timit = timit.remove_columns(["phonetic_detail", "word_detail", "dialect_region", "id", "sentence_type", "speaker_id"])

import re
chars_to_ignore_regex = r'[,?.!-;:"]'

def remove_special_characters(batch):
    batch["text"] = re.sub(chars_to_ignore_regex, '', batch["text"]).lower()
    return batch

timit = timit.map(remove_special_characters)

for _ in range(10):
    print(timit["train"][random.randint(0, len(timit["train"])-1)]["text"])

print(timit)

# Save the dataset
timit.save_to_disk("timit")

# Load timit again
