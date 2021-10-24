# yuminzhe-Prediction-of-peptide-detectability-based-on-CapsNet-and-CBAM-module
We proposes a model combining CapsNet and CBAM attention module to predict the detectability of peptides based on MS experiments. It constructs the residue conic coordinates feature, and combine amino acid and dipeptide components, sequence embedding to generate peptide chain features for the proposed network. As for these features, they are divided into biological feature and sequence feature by separate input-ting to reduce the influence among these features. When using the Capsnet network, it reduces the impact of data loss in the pooling layer after convolution. In addition, a CBAM attention module is added to assign weights to channels and spaces to learn important features. The experimental results verify the effectiveness of the proposed method, and it can be used as a valid supplementary method for peptide detectability prediction and applied in proteomics and other fields.



## Data availability
In the "data" folder, we provide the all the datasets used in the paper. If you want to use them, please download them first.


## Model
In the "model" folder, “params.pkl” is the best trained model (params1.pkl, Params2.pkl, …, Params10.pkl are the models for the ten-fold cross-validation). If you want to use them, please download them first.


## Usage
You can upload the file CapseNet.ipynb to Colab (http://colab.research.google.com) for training and testing in the GPU environment.

## Test personal data set
We also provide scripts for testing different data sets.First you have to execute the following command to install the environment：
```
pip install -r requirements.txt
```
After that, the user can test different data sets through the following commands：
```
python sequence_test.py --test=test.csv --model=params1.pkl --result=result.txt
```
Where test is the location of the data set, and the format of the data set needs to be the same as the format of the data set in the data folder.Model is the location of the trained model.Result is the location where the result is saved.


