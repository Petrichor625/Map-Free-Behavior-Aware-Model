# HiVT:xxxx

This repository contains the official implementation of xxxxx

## Background



## Note

[2023.4.17] If you install Argoverse-api encounter the problem "[ERROR] Cannot make wheel for opencv"

## Install

The model install in Ubuntu 20.04, cuda11.7

**1.Clone this repository**: clone our model use following code 

```
git clone https://github.com/xxxxxxxxxx
cd HiVT
```

**2.Implemenation Environment**: The model is implemented by using Pytorch. We share our anaconda environment in the folder 'environments',then use this command to implement your environment.

```
cd environments
conda env create -f environment.yml
```

If this command cannot implement your conda environment well, try to install again the specific package separately.

```
conda create -n HiVT python=3.8
conda activate HiVT
conda install pytorch==1.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
conda install pytorch-geometric==1.7.2 -c rusty1s -c conda-forge
conda install pytorch-lightning==1.5.2 -c conda-forge
```

**3.download Argoverse Dataset 1.1**

.Before starting the training process, you need to prepare the dataset and follow the steps below.

Argoverse provide both the full dataset and the sample version of the Argoverse dataset 1.1 for testing purposes.We will train and tese our models in the Argoverse.Please download **Argoverse Motion Forecasting v1.1** and **HD maps** files in [Argoverse 1.1](https://www.argoverse.org/av1.html#forecasting-link)

Than you can extract your dataset file by following command.Put all you dataset in the file named `ArgoverseDataset`

```
cd YourPath/ArgoverseDataset
tar xvf forecasting_train_v1.1.tar.gz
tar xvf forecasting_val_v1.1.tar.gz
tar xvf forecasting_test_v1.1.tar.gz
```

**4.Download Argoverse API **

Download argoverse-api (1.0) in another folder (out of this directory):

```
git clone https://github.com/argoai/argoverse-api.git
```

Extract the HD map.zip into the root directory of the repo. 

```
cd YourPath /Argoverse-api
tar xf hd_maps.tar.gz
```

Your directory structure should look something like this:

```
argoverse-api
└── argoverse
    └── data_loading
    └── evaluation
    └── map_representation
    └── utils
    └── visualization
└── map_files
...
└── license
```

Go to the argoverse-api folder and install the requirements of setup.py.

```
cd YourPath/argoverse-api
pip install -e .
```



## Train Model

In this section, we will explain how to train the model.

Please keep your model in file named `models` and change your hyperparameter in models.py

To train the model from scratch, run the followings.

**Attention**: It is worth noting that the initial processing of the dataset might take a significant amount of time, typically around 2 to 3 hours, depending on the size and complexity of the dataset. This processing time is necessary to ensure the data is properly prepared for training.

```
python train.py --root /YourPath/dataset --embed_dim 128
```

The training logs will keep in the file name `Train_logs`,you can use tensorboard to cheek your training process.

```
tensorboard --logdir /YourPath/Train_log/yourmodel
```



## Evaluation

This step helps you assess how well the model generalizes to unseen data and allows you to make any necessary adjustments or improvements. If the model does not meet your desired performance criteria, you can iterate on the training process by adjusting hyperparameters, modifying the model architecture, or collecting additional data.

Once the model has been trained, it is important to evaluate its performance on a separate validation or test set. This step helps assess how well the model generalizes to unseen data. You can choose to evaluate the model using either the validation set (val) or the test set. By specifying the desired mode (e.g., val or test), a .h5 file will be generated with the evaluation results.

To evaluate the model on the validation set, run the following command:

```
python evaluate.py --mode=val --root /path/to/dataset_root/ --ckpt_path /path/to/your_checkpoint.ckpt
```

To evaluate the model on the test set, run the following command:

```
python evaluate.py --mode=test --root /path/to/dataset_root/ --ckpt_path /path/to/your_checkpoint.ckpt
```



## Qualitative results

We are preparing a script for generating these visualizations:

 ````
 ....
 ````



![img](https://github.com/ZikangZhou/HiVT/raw/main/assets/visualization.png)



## Conclusion

Training a model involves dataset preparation, data processing, model training, evaluation, and iteration for improvement. The initial dataset processing step might take a considerable amount of time, typically around 2 to 3 hours. However, this step is crucial for ensuring the data is properly prepared for training. Once the dataset is processed, you can proceed with training the model, evaluate its performance using either the validation or test set, and make iterative improvements as needed.

## Citation

If you use our source code, please cite:

```
```

