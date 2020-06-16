### Tweet-Sentiment-Extraction Challenge

#### Requirements

Python >= 3.6<br>
TensorFlow-GPU >= 2.0<br>

To install required third-party Python packages, run `pip install -r requirements.txt`


#### Modeling

To fit and predict with the transformer model(s), first run `chmod +x run.sh` (only has to be run once), then `FOLD=0 MODEL=xlnet ./run.sh` to run the xlnet transformer<br>

Highest scoring model weights will be saved in `src/tweet-sentiment-extraction/weights/`. See `infer.py` on how to make predictions with saved model weights.
