# WEP-Word-Embedding
Word Embedding with Neural Probabilistic Prior,  Ren et al., SDM 2024

This is the code for the paper 'Word Embedding with Neural Probabilistic Prior', Ren et al., SDM 2024.


## Run the code

1. Install packages
```
pip3 install -r requirements.txt
```

2. Download the dataset:
```
pip install gdown
gdown --id 1iFpuKFpDnXCD9QpUw8wStG3ndKl7-KwX -O data.zip
unzip data.zip
rm data.zip
```
3. Run WEPSyn; example command line: 
```
python3  WEPSyn.py -name test_embeddings -alpha 1.0 -gpu 1 -dump -embed_dim 300 -batch 256
```
4. Run WEPSem; example command line: 
```
nohup python3 WEPSem.py -embed ./embeddings/pretrained_embed  -semantic synonyms -embed_dim 300 -alpha 0.001  -name fine_tuned_embeddings -dump -gpu 5
```


## To cite the paper:
```
@inproceedings{ren2024word,
  title={Word Embedding with Neural Probabilistic Prior},
  author={Ren, Shaogang and Li, Dingcheng and Li, Ping},
  booktitle={Proceedings of the 2024 SIAM International Conference on Data Mining (SDM)},
  pages={896--904},
  year={2024},
  organization={SIAM}
}
```

The package was developed based on the implementation of 'Incorporating Syntactic and Semantic Information in Word Embeddings using Graph Convolutional Networks', Vashishth et al., ACL'19.
(https://github.com/malllabiisc/WordGCN)

