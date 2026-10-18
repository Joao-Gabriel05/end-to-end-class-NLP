# End-to-end IMDB classifier: experiments

Activity from [Tokenizers, Classification, and Visualization](https://tiagoft.github.io/nlp_course/language_models/03-end_to_end_neural_networks/).
Everything is in `experiments.ipynb`. Results are in `results/*.csv` and plots in `figures/`.

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace experiments.ipynb   # ~20 min on a laptop CPU; downloads IMDB to data/
```

**Setup:** SentencePiece (trained on the run's training texts) → `nn.Embedding` → masked mean → `nn.Linear`.
Training uses BCE-with-logits, Adam (lr 1e-2), batch 64, 20 epochs and max length 256. Evaluation uses a fixed set of 5000 reviews
from the IMDB test split. Baseline: **vocab 4000, 5000 training reviews, embedding dim 32**. Each sweep
changes one parameter, with 3 seeds per value. The tables show eval accuracy as mean (± std).

## 1. Vocabulary size

| vocab | final acc | best-epoch acc | params |
|---:|---:|---:|---:|
| 100 | 0.613 ± .006 | 0.618 | 3k |
| 500 | 0.730 ± .002 | 0.734 | 16k |
| 1000 | 0.763 ± .004 | 0.772 | 32k |
| 4000 | 0.787 ± .004 | 0.814 | 128k |
| 16000 | **0.812 ± .002** | **0.825** | 512k |

Bigger vocabularies help steadily. With 100 pieces the tokens are mostly characters and short fragments that
carry no sentiment, so averaging them loses almost everything. Whole words like *awful* and *brilliant* only
become single tokens with a large vocabulary. The catch is that larger vocabularies reach their best epoch sooner (~4)
and overfit after that.

## 2. Dataset size

| train reviews | final acc | best-epoch acc | best epoch |
|---:|---:|---:|---:|
| 100 | 0.573 ± .034 | 0.573 | 20 |
| 500 | 0.720 ± .003 | 0.722 | 17 |
| 2000 | 0.778 ± .002 | 0.786 | 7 |
| 5000 | 0.787 ± .004 | 0.814 | 4 |
| 25000 | **0.815 ± .003** | **0.837** | 2 |

This setting is the main reason the class notebook got near-chance results: 100 reviews is roughly
the class's setting, and it gets 0.57 here as well. More data gives the largest single gain and also makes the model overfit less. With
25k reviews the gap between train and eval loss is the smallest of any run.

## 3. Embedding dimension

| dim | final acc | best-epoch acc | best epoch | params |
|---:|---:|---:|---:|---:|
| 2 | **0.811 ± .004** | 0.813 | 16 | 8k |
| 8 | 0.797 ± .005 | **0.814** | 8 | 32k |
| 32 | 0.787 ± .004 | 0.814 | 4 | 128k |
| 128 | 0.789 ± .002 | 0.811 | 4 | 512k |
| 512 | 0.785 ± .006 | 0.809 | 3 | 2M |

Dimension barely matters. Best-epoch accuracy stays at about 0.81 for every value, because a mean-pooled bag of tokens followed by a linear layer
behaves like a linear model on token counts, and that needs very little capacity. Larger dimensions only overfit faster:
at 20 epochs, `dim=2` ends up with the best final accuracy. In `figures/embedding_dim_embeddings.png` the
**train** embeddings split into two clean clusters for every dimension of 8 or more, but the **eval** embeddings overlap. That
mismatch is the overfitting signature the class described.

## Extension: sequence models (10 epochs, 1 seed)

| model | final acc | best acc | params | time |
|---|---:|---:|---:|---:|
| mean | **0.801** | **0.817** | 128k | 2 s |
| BiRNN | 0.772 | 0.778 | 141k | 62 s |
| BiGRU | 0.771 | 0.788 | 166k | 151 s |
| BiLSTM | 0.781 | 0.781 | 178k | 192 s |

With only 5000 reviews and 10 epochs, the recurrent models do **not** beat the mean, and they run 30–100× slower.
GRU and LSTM *can* use word order (for example "not good"), but they need more data or epochs, plus tuning such as
dropout and pretrained embeddings, before that advantage shows up.

## Conclusions

1. **Data size and vocabulary size matter. Embedding dimension doesn't.** The best single changes were 25k reviews (0.84 at best epoch)
   and a 16k vocabulary (0.83 at best epoch), compared with ~0.55 in the class notebook.
2. **Overfitting is the main limit.** Eval loss bottoms out after 2–5 epochs and then climbs while
   train loss keeps falling toward 0. Early stopping, or picking the epoch by eval loss, is worth about +2–3 points.
3. Simple fixes help too: a tokenizer trained on in-domain text and a masked mean that ignores `<PAD>`.
4. A sequence model is not a free win at this scale.

Caveat: the "best epoch" numbers are picked using the eval set itself, so they are optimistic. A proper setup
would hold out a validation split for that choice.
