## Environment

Python 3.6.5 & Tensorflow > 1.3

## Data

All data, including data collection and relation production procedure are uploaded here:
链接：https://pan.baidu.com/s/17dIzlqUNrSzF5Zt6CLHGDg?pwd=ints 
提取码：ints

### Sequential Data

Raw data: files under the google_finance folder are the historical (30 years) End-of-day data (i.e., open, high, low, close prices and trading volume) of more than 8,000 stocks traded in US stock market collected from Google Finance.

Processed data: folder '2013-01-01', '2014-01-01' are the dataset used to conducted experiments in our report


### Industry Relation

Under the sector_industry folder, there are row relation files storing the industry relations between stocks in NASDAQ and NYSE.

### Wiki Relation

Under the wikidata folder, there are row relation files storing the Wiki relations between stocks in NASDAQ and NYSE.

## Code

### Pre-processing

| Script | Function |
| :-----------: | :-----------: |
| eod.py | To generate features from raw End-of-day data |
| sector_industry.py | Generate binary encoding of industry relation |
| wikidata.py | Generate binary encoding of Wiki relation |

### Training
| Script | Function |
| :-----------: | :-----------: |
| rank_lstm.py | Train a model of Rank_LSTM |
| relation_rank_lstm.py | Train a model of Relational Stock Ranking |


## Run

To repeat the project, it requires to unzip the project and process files with order: 

```
1. extrat the data files in the correct path 
2. run 

```

## Cite

This project has based on ideas of the following papers: 
```
[1] Jonathan L. Elsas; Susan T. Dumais;  "Leveraging Temporal Dynamics of Document Content in Relevance Ranking",  2010.  (IF: 4)
[2] Swati Rallapalli; Lili Qiu; Yin Zhang; Yi-Chao Chen;  "Exploiting Temporal Stability And Low-rank Structure For Localization In Mobile Networks",   MOBICOM,  2010.  (IF: 4)
[3] Ram Babu Roy; Uttam Kumar Sarkar;  "Identifying Influential Stock Indices from Global Stock Markets: A Social Network Analysis Approach",  2011.  (IF: 3)
[4] Jeffrey Jestes; Jeff M. Phillips; Feifei Li; Mingwang Tang;  "Ranking Large Temporal Data",   ARXIV-CS.DB,  2012.  (IF: 3)
[5] Basura Fernando; Efstratios Gavves; Jose Oramas; Amir Ghodrati; Tinne Tuytelaars;  "Rank Pooling For Action Recognition",   ARXIV-CS.CV,  2015.  (IF: 5)
[6] Xiyang Dai; Bharat Singh; Guyue Zhang; Larry S. Davis; Yan Qiu Chen;  "Temporal Context Network For Activity Localization In Videos",   ARXIV-CS.CV,  2017.  (IF: 5)
[7] Fuli Feng; Xiangnan He; Xiang Wang; Cheng Luo; Yiqun Liu; Tat-Seng Chua;  "Temporal Relational Ranking For Stock Prediction",   ARXIV-CS.CE,  2018.  (IF: 4)
[8] Mostafa Shabani; Alexandros Iosifidis;  "Low-Rank Temporal Attention-Augmented Bilinear Network for Financial Time-series Forecasting",   2020 IEEE SYMPOSIUM SERIES ON COMPUTATIONAL INTELLIGENCE ...,  2020.
[9] Ramit Sawhney; Shivam Agarwal; Arnav Wadhwa; Tyler Derr; Rajiv Ratn Shah;  "Stock Selection Via Spatiotemporal Hypergraph Attention Network: A Learning to Rank Approach",   AAAI,  2021.  (IF: 3)
[10] Ramit Sawhney; Shivam Agarwal; Arnav Wadhwa; Rajiv Shah;  "Exploring The Scale-Free Nature of Stock Markets: Hyperbolic Graph Learning for Algorithmic Trading",   WWW,  2021.  (IF: 3)
```

