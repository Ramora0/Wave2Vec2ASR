# Base

The point of this project is to implement a compression scheme into audio ASR models to improve memory and performance, and down the line align representations across multiple modalities in multimodal llms.

# Structure

Our best performing setup is explained here.

### Model

We use whisper small w/ 244m parameters for quick testing.

### Position

Our boundary predictor + downsampling mechanism goes after the convolutional layers (there are two, working on the spectrogram of the audio; first is stride 1, second is stride 2) and before the positional encoding and transformer layers. This allows us to fully maximize the squared cost of attention, since we downsample greatly before any layer.

### Boundary Predictor

We use a boundary predictor proposed by HNet; we train a query and key matrix along with an MLP, then, for every pair of states, we dot product their Query and Keys, and treat that value as a pseudo boundary probability.

### Boundary Sampling

We use Gumbel-Sigmoid, a Relaxed Bernoulli Distribution, and a Straight through estimator to retain boundary differentiability, since each segment is ostensibly a hard decision boundary.

### Downsampling

For downsampling, we use an attention pooling mechanism; each sequence defined by the boundaries is key projected, then dot product'd with a single learned query vector for all segments, which is then pooled like normal.

### Compression Rate

We target a variable compression rate by using a binomial loss of # of boundaries, # of target boundaries, and # of positions. For # of target boundaries, we use either the number of phonemes or syllables in the selected transcription, achieving compression rates of roughly 5x and 13.5x respectively.

# Results

### Performance

With this setup we get 5% WER with 5x phoneme compression and 7% WER with 14x syllable level compression.

### Loss Dynamics

For syllable level compression, we see an initial collapse of loss, followed by minimal progress, then another collapse of the loss roughly 3-4 epochs in to our 5 epoch training.

### Annealing

We anneal the temperature of the gumbel sigmoid distribtuion over time from 1 to 0.

# History / Experiments

Chronologically, most recent first:

### Uniformity Loss

In phoneme level compression, the percentage of boundaries directly next to each other is ~100%, suggesting the model is wasting space. To combat this, I tried adding two loss terms on seperate runs; a repulsion loss (1d kernel 3 conv on boundary decisions) and a uniformity loss (take random subsets of the data and enforce target compression there)
The repulsion loss initially improved performance early in training, but lead to the exact same result by the end. The uniformity loss started slightly higher, but also converged to the exact same final WER.

### Temperature Scheduling

Just did a quick run to confirm the temperature scheduling was necessary, and was confirmed by an 8.5% WER without it.

### Other Boundary Predictor

We tried replacing the q-k boundary predictor with a single MLP to predict probability, and while training did initially look similar, it never exhibitted that second loss collapse that the other did and didn't seem to work.

### Attention Pooling

Previously we used straight mean pooling, but when replacing with attention we see particularly strong results with syllable level compression, which sees a noticeably huge loss collapse after the 3-4th epoch all the way down to ~7%, from all the way at 20%. We did not see nearly as improved performance with phoneme level compression, going from ~7% to ~5%.
