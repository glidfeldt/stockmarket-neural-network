# stockmarket-neural-network

## To run Reccurent Neural network
```bash
python RNNMulti.py
```

## Relevant variables:
* Global variables
  * BATCH_SIZE - lice the data into batches
  * BUFFER_SIZE - Buffer size to shuffle the dataset (TF data is designed to work with possibly infinite sequences, so it doesn't attempt to shuffle the entire sequence in memory. Instead, it maintains a buffer in which it shuffles elements).
  * EVALUATION_INTERVAL
  * EPOCHS - Iterate over the dataset
  * past_history - Days to analyze
  * future_target - Days to predict
  * STEP = 6
* Local variables:
  * filename - stock to be used
  * TRAIN_SPLIT - where to split the dataset training and testing
