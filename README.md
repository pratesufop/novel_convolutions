# Novel Convolutions


This repository implements some convolutional layers tailored to specific applications. Particularly, we implemented the following CNNs:

[Central Difference Convolution (CDC)](https://arxiv.org/abs/2003.04092)

[Median Pixel Difference Convolution](https://bmvc2021-virtualconference.com/conference/papers/paper_0145.html)\\

[LBPConv](https://arxiv.org/abs/1608.06049)

[ConstrainedCNN](https://ieeexplore.ieee.org/document/8335799)


We validate these convolutions using the MNIST dataset and a simple CNN network with three layers with 16, 16 and 32 filters. Thus, we replace the first convolutional layer with one the novel convolutions. The obtained experimental results are presented in the Table 1. 


Table 1. Experimental results for different types of convolutions.

| Approach       | #parameters | Acc.  |
|----------------|-------------|-------|
| CDC            | 22,890      | 98.94 |
| ConstrainedCNN | 22,890      | 98.87 |
| LBPConv        | 23,642      | 98.97 |
| MeDiConv       | 22,986      | 98.68 |



To run the code use the following command with the options (CDC, ConstrainedCNN, LBPConv, MeDiConv). 


```python
python main.py --conv=option
```
