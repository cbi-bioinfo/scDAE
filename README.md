# Cell type classification via representation learning based on denoising autoencoder for single-cell RNA sequencing
scDAE is a DNN model for single cell subtype identification combined with representative feature extraction by multilayered denoising autoencoder. The feature sets were learned by the denoising autoencoder and were further tuned by fully connected layers using a softmax classifier. scDAE can efficiently predict cell type on a well-trained representation learning model, which may help to improve precision of single cell analysis.

![Figure](https://github.com/cbi-bioinfo/scDAE/blob/master/celltype_classification_v6.png?raw=true)

## Requirements
* Tensorflow (>= 1.8.0)
* Python (>= 2.7)
* Python packages : numpy, pandas

## Usage
Clone the repository or download source code files and prepare scRNA-seq dataset.

1. Edit **"run_scDAE.sh"** file having scRNA-seq dataset files for model training and testing with cell type annotations for each sample. Modify each variable values in the bash file with filename for your own dataset. Each file shoudl contain the header and follow the format described as follows :

- ```train_X, test_X``` : File with a matrix or a data frame containing gene expression of features for model training and testing, where each row and column represent **sample** and **gene**, respectively. Example for dataset format is provided below.

```
A1BG,A1CF,A2M,A2ML1,...,ZZEF1,ZZZ3
6.9056,15.2654,10.4164,20.8916,...,0.0,10.3074
15.991,5.8096,0.0,45.5589,...,0.0,28.9703
10.2477,10.2712,0.0,0.0,...,0.0,17.2436
...
```

- ```train_Y, test_Y``` : File with a matrix or a data frame contatining cell type annotation for each sample, where each row represent **sample**. Cell type names used for training and testing should be included and users should label each cell type as 1 and 0 for others in the same order in training dataset to be matched. Example for data format is described below.

```
alpha,beta,delta,gamma
1,0,0,0
0,1,0,0
0,0,0,1
...
```

2. Use **"run_scDAE.sh"** to classify cell types in gene expression dataset based on single-cell RNA sequencing.

3. You will get an output **"result_for_test_dataset.csv"** with classified cell types for test dataset.

## Identification of a new cell subtype
1. Use **"get_probability_by_softmax.sh"** to output the probabilities for each cell subtype estimated through the softmax function in the classification step from scDAE.

2. Label the cell "new cell subtype" if the highest probability is lower than 0.95, otherwise classify it as a predicted cell type. 

## Contact
If you have any question or problem, please send an email to **miniymay AT sookmyung.ac.kr**
