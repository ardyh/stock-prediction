Replicating the Papers:
1. Main paper: [Deep Attentive Learning for Stock Movement Prediction From Social Media Text and Company Correlations](https://aclanthology.org/2020.emnlp-main.676.pdf)
2. Data: [Stock Movement Prediction from Tweets and Historical Prices](https://aclanthology.org/P18-1183.pdf)
3. Data: [Temporal Relational Ranking for Stock Prediction](https://arxiv.org/pdf/1809.09441)

Our main google drive: 
https://drive.google.com/drive/folders/1mukwh3Fuldy5bRCQZQGmh0G1ZgaJe1yx?usp=sharing

You should be able to run the colab there (MAN-SF Sandbox.ipynb)

Code we wrote and data we generated ourselves
- `improv_data/` | the dataset we generated
- `gen_graph.ipynb` | for generating the graph on phase 1
- `process_data_improv.ipynb` | all data processing for phase 2 (prices, tweets, graph) 
- `process_data_stocknet-code.ipynb` | data processing phase 1 (prices, tweets) 
- `run_datapipe.ipynb` | running StockNet's train-test generator pipeline (prices, tweets)

Code we improved from the original paper's codebase:
- `train.py` | added feature to save model weight and per-epoch data 
- `train_local`
- `model.py` | added layernorm and residual layer
