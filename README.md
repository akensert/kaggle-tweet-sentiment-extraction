### tweet-sentiment-extraction


#### Requirements

Python >= 3.6
TensorFlow-GPU >= 2.0

To install required python packages, run:

```pip install -r requirements.txxt
```


#### Modeling

To fit and predict with the transformer model(s), first run `chmod +x run.sh` (only has to be run once), then `FOLD=0 REPL=0 MODEL=xlnet ./run.sh`<br>

Highest scoring model will be saved in `src/weights/`
