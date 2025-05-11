import json
from datasets import load_from_disk
import matplotlib.pyplot as plt
from tqdm import tqdm

def ctc(labels):
    """
    Calculate the minimum CTC length for a given labels.
    """
    count = 0
    # adjacent_equals = sum(1 for i in range(len(labels) - 1) if labels[i] == labels[i+1])
    count += sum(1 for i in range(len(labels)) if labels[i] != 10)
    return count + len(labels)

def calculate_ctc_ratio(example):
    """Calculates the ratio of original length to CTC length for an example."""
    labels = example['labels']
    ctc_length = ctc(labels)
    # Avoid division by zero if ctc_length is somehow zero

    inputs = len(example['input_values']) / 16000 * 50

    ratio = ctc_length / inputs
    return {"ctc_ratio": ratio}

if __name__ == "__main__":
    dataset = load_from_disk("./data/librispeech_processed")
    # Use the map function to apply the calculation across the dataset
    subset_dataset = dataset["train.clean.100"].select(range(2998))
    processed_dataset = subset_dataset.map(calculate_ctc_ratio, num_proc=4) # Adjust num_proc based on your CPU cores

    ctc_lengths = processed_dataset["ctc_ratio"]
    
    # Print the mean and standard deviation of the CTC lengths
    mean_ctc_length = sum(ctc_lengths) / len(ctc_lengths)
    std_ctc_length = (sum((x - mean_ctc_length) ** 2 for x in ctc_lengths) / len(ctc_lengths)) ** 0.5
    print(f"Mean: {mean_ctc_length:.2f}")
    print(f"SD: {std_ctc_length:.2f}")
        
    plt.figure(figsize=(10, 6))
    plt.hist(ctc_lengths, bins=50, color='blue', alpha=0.7)
    plt.title('Distribution of Original Length / CTC Length Ratio')
    plt.xlabel('Ratio (Original Length / CTC Length)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()