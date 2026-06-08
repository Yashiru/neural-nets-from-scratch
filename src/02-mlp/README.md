# 02 | MLP

A character-level language model with a **multi-layer perceptron**, following
Bengio et al. (2003). Where the bigram only looked one letter back, here we feed
the model a window of several letters and let it learn its own features - through
**embeddings** and a hidden layer - instead of reading them off a count matrix.

The goal is to lift the bigram's hard limit (one letter of context) and, along
the way, meet the building blocks every later experiment reuses: embeddings, a
hidden layer, minibatch training, a learning-rate schedule, and an honest
train/dev/test split.

## The problem

Same ~32,000 names (`data/names.txt`), same 27-symbol vocabulary (`a`–`z` plus
the boundary token `.`). But now each prediction sees the **last 3 characters**
of context instead of one:

```
.emma  ->  ...->e   ..e->m   .em->m   emm->a   mma->.
```

A bigram has to summarize everything it knows in a `27×27` table. Widen the
context to 3 characters and a count table would need `27³` rows - most of them
never seen. The MLP sidesteps this: it learns a dense representation of context
rather than memorizing every combination.

## The model

```
context (3 chars)  ->  embed  ->  concat  ->  tanh hidden  ->  logits  ->  softmax
```

- **Embedding.** Each character is mapped to a learned vector of size `10`
  (matrix `C`, shape `27×10`). Similar letters can end up with similar vectors -
  something a one-hot bigram can't express.
- **Concatenate.** The 3 context embeddings are flattened into one `30`-d input.
- **Hidden layer.** A linear layer `W1, b1` followed by `tanh` gives `200`
  hidden features.
- **Output.** A second linear layer `W2, b2` produces 27 *logits*, turned into a
  probability distribution by **softmax**.

That's **11,897 parameters** in total, trained end to end by gradient descent.

![MLP results in the terminal](assets/results.png)

The script prints the whole run: dataset and model summary, a live training
progress bar, the loss curve, the held-out evaluation, a sampling trace, and the
generated names.

## Training

```
sample a minibatch  ->  forward  ->  cross-entropy  ->  backward  ->  SGD step
```

- We shuffle the names and split **80 / 10 / 10** into train / dev / test
  (182k / 23k / 23k examples). The dev and test sets are never trained on, so
  their loss is an honest measure of generalization.
- Each step trains on a random **minibatch of 32** examples - fast, noisy
  gradients that are good enough to make progress.
- The loss is the **cross-entropy** between the predicted distribution and the
  true next character (the same negative log-likelihood as the bigram).
- The **learning rate** starts at `0.1` and drops to `0.01` after 60% of the
  50,000 steps, to settle into a finer minimum at the end.

## Results

| split | cross-entropy (nats) |
| --- | --- |
| train | ≈ 2.22 |
| dev   | ≈ 2.25 |
| test  | ≈ 2.25 |

Dev and test track train closely, so the model **generalizes** rather than
memorizes. The loss is clearly below the bigram's, and the samples read more like
plausible names:

```
rumneel - mictai - jermina - esvala - addi - samarlin - tamya - caraelie - tenley
```

Still imperfect - 3 characters of context is short - but a real step up from the
one-letter bigram.

## Running the experiment

From the repo root:

```bash
python3 src/02-mlp/main.py
```

It loads the data, trains for 50k steps (~15s on CPU) with a live progress bar,
then prints the evaluation and a batch of sampled names.

## What it teaches

- **Embeddings** beat one-hot: the model learns a dense, shared representation of
  characters instead of a row per symbol.
- Why we widen context with an MLP instead of a bigger count table - the table
  grows exponentially, the network doesn't.
- The standard training loop: **minibatch SGD**, a **learning-rate schedule**,
  and a **train/dev/test split** to measure generalization honestly.
- A hint of what's next: the initial loss is ~`27`, far higher than the ~`3.3`
  you'd expect from a uniform guess. That spike means the network starts
  *overconfident* - a symptom of poor **initialization**. Fixing that (and
  keeping activations healthy with **batch norm**) is exactly the subject of the
  `03-activations-batchnorm` experiment.
