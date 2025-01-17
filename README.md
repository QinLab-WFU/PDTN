# PDTN
This is our TensorFlow implementation for the paper:
Neurocomputing'25 [PDF]
Please cite our paper if you use the code:
```bibtex
@article{ge2025personalized,
  title={Personalized Dual Transformer Network for sequential recommendation},
  author={Ge, Meiling and Wang, Chengduan and Qin, Xueyang and Dai, Jiangyan and Huang, Lei and Qin, Qibing and Zhang, Wenfeng},
  journal={Neurocomputing},
  pages={129244},
  year={2025},
  publisher={Elsevier}
}```

## Paper Abstract
Sequential Recommendation (SR) seeks to anticipate the item that users will interact with at the next moment, utilizing their historical sequences of interactions. Its core task is to mine users’
interests. Most existing transformer-based models focus on extracting user’s interests by leveraging local context information from individual interaction sequences, ignoring the user’s personalized
characteristics. Furthermore, when the sequence length is significantly shorter than the specified threshold, these transformer-based models encounter challenges associated with the cold-start issue.
To tackle the mentioned issues, we propose a novel transformer-based sequential model, named Personalized Dual Transformer Network (PDTN), that extends the length of the user’s sequences
by leveraging the user’s historical behaviors to search for relevant items. Specifically, a personalized feature extraction module is proposed to extract characteristics of both the user and the user’s historical behaviors. Then, a Dual Transformer is designed to retrieve a user’s relevant historical behaviors, which can increase the length of the user’s historical interaction sequences and fully extract the user’s interests. Furthermore, Stochastic Shared Embeddings (SSE) regularization is developed into a Transformer-based sequential model to reduce overfitting and errors in the processing of training. Extensive experiments are conducted to compare PDTN with current methods on four publicly available real-world datasets for prediction tasks. The results consistently demonstrate that PDTN outperforms advanced sequential recommendation methods. The source code is publicly available at https://github.com/QinLab-WFU/PDTN.

## Code introduction
The code is implemented based on Tensorflow version of [SASRec](https://github.com/kang205/SASRec).

## Environment Setup
The code is tested under a Linux desktop (NVIDIA RTX 3090 GPU) with TensorFlow 1.12 and Python 3.6.
Create the requirement with the requirements.txt

## Datasets
We use the Amazon Review datasets Beauty and Cell_Phones_and_Accessories. The data split is done in the
leave-one-out setting. Make sure you download the datasets from the [link](https://jmcauley.ucsd.edu/data/amazon/).

### Data Preprocessing
Use the DataProcessing.py under the data/, and make sure you change the DATASET variable
value to your dataset name, then you run:
```
python DataProcessing.py
```
You will find the processed dataset in the directory with the name of your input dataset.


## Baby Dataset Pre-training and Prediction
### Reversely Pre-training and Short Sequence Augmentation
Pre-train the model and output 20 items for sequences with length <= 20.
```
python main.py --dataset=Beauty --train_dir=default --lr=0.001 --hidden_units=128 --maxlen=100 --dropout_rate=0.7 --num_blocks=2 --l2_emb=0.0 --num_heads=4 --evalnegsample 100 --reversed 1 --reversed_gen_num 20 --M 20
```
### Next-Item Prediction with Reversed-Pre-Trained Model and Augmented dataset
```
python main.py --dataset=Beauty --train_dir=default --lr=0.001 --hidden_units=128 --maxlen=100 --dropout_rate=0.7 --num_blocks=2 --l2_emb=0.0 --num_heads=4 --evalnegsample 100 --reversed_pretrain 1 --aug_traindata 15 --M 18 --threshold_user 0.08 --threshold_item 0.9
```
--threshold_user 0.08 --threshold_item 0.9

## Cell_Phones_and_Accessories Dataset Pre-training and Prediction
### Reversely Pre-training and Short Sequence Augmentation
Pre-train the model and output 20 items for sequences with length <= 20.
```
python main.py --dataset=Cell_Phones_and_Accessories --train_dir=default --lr=0.001 --maxlen=100 --dropout_rate=0.5 --num_blocks=2 --l2_emb=0.0 --num_heads=2 --evalnegsample 100 --reversed 1 --reversed_gen_num 20 --M 20
```
### Next-Item Prediction with Reversed-Pre-Trained Model and Augmented dataset
```
python main.py --dataset=Cell_Phones_and_Accessories --train_dir=default --lr=0.001 --maxlen=100 --dropout_rate=0.5 --num_blocks=2 --l2_emb=0.0 --num_heads=2 --evalnegsample 100 --reversed_pretrain 1  --aug_traindata 17 --M 18
```


