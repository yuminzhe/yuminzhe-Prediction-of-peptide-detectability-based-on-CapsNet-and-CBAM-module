# yuminzhe-Prediction-of-peptide-detectability-based-on-CapsNet-and-CBAM-module
We proposes a model combining CapsNet and CBAM attention module to predict the detectability of peptides based on MS experiments. It constructs the residue conic coordinates feature, and combine amino acid and dipeptide components, sequence embedding to generate peptide chain features for the proposed network. As for these features, they are divided into biological feature and sequence feature by separate input-ting to reduce the influence among these features. When using the Capsnet network, it reduces the impact of data loss in the pooling layer after convolution. In addition, a CBAM attention module is added to assign weights to channels and spaces to learn important features. The experimental results verify the effectiveness of the proposed method, and it can be used as a valid supplementary method for peptide detectability prediction and applied in proteomics and other fields.



## Data availability
In the "data" folder, we provide the all the datasets used in the paper. If you want to use them, please download them first.


## Model
In the "model" folder, “params.pkl” is the best trained model (params1.pkl, Params2.pkl, …, Params10.pkl are the models for the ten-fold cross-validation). If you want to use them, please download them first.


## Usage
You can upload the file CapseNet.ipynb to Colab (http://colab.research.google.com) for training and testing in the GPU environment.

## Test personal data set
We also provide scripts for testing different data sets.We give an example to show the execution process：</br>
(1)Prepare the peptide chain file (the file extension is .csv), and the format of peptides in this file is shown as</br>
![csv](https://github.com/yuminzhe/yuminzhe-Prediction-of-peptide-detectability-based-on-CapsNet-and-CBAM-module/blob/main/figure1.png)<br>

(2)Then, Download our trained model, the website is https://github.com/yuminzhe/yuminzhe-Prediction-of-peptide-detectability-based-on-CapsNet-and-CBAM-module/tree/main/model </br>
(3)Execute the following command to realize the detectability prediction of peptides：</br>

```bash
python sequence_test.py --test=test.csv --model=params.pkl --result=result.txt
```
--test represents the directory where the input peptide chain file is located, --model is the directory where the trained model is located, and --result sets the location of the result output.</br>
The running result is shown as：<br>
![result1](https://github.com/yuminzhe/yuminzhe-Prediction-of-peptide-detectability-based-on-CapsNet-and-CBAM-module/blob/main/figure2.jpg)<br>
The output of “result.txt” (saving the detectability prediction result of peptides) is shown as</br>
 ![result2](https://github.com/yuminzhe/yuminzhe-Prediction-of-peptide-detectability-based-on-CapsNet-and-CBAM-module/blob/main/figure3.jpg)<br>
Ten numbers respectively represent the results of the detectability of the predicted peptides from ten sequences inputted above, where 0 represents undetectable and 1 represents detectable.

