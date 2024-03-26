# SentimentAnalysis
Sentiment analysis of tweets

## Data
Data has been pulled from [Kable link](https://www.kaggle.com/datasets/kazanova/sentiment140) which is a 277MB csv file (placed in data folder)

### Columns 

| Column name | Description | Value example |
| ----------- | ----------- | ------------- |
| target      | the polarity of the tweet | (0 = negative, 2 = neutral, 4 = positive) |
| ids         | The id of the tweet |  2087 |
| date        | the date of the tweet | Sat May 16 23:58:44 UTC 2009 |
| flag        | The query (lyx). If there is no query, then this value is NO_QUERY.|  |
| user        | the user that tweeted | robotickilldozr |
| text        | the text of the tweet | Lyx is cool |

## Hardware

| Harware | Type |
| --- | ---- |
| CPU | 11th Gen Intel Core i7 - 11800H 2.30 GHz |
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU |
| Memory | 64 GB |

## Libraries being used
* tensorflow is being used to train model
* wordcloud, matplotlib and seaborn is used to visualize data and model
* numpy and pandas

## Installation steps
There are lot of dependencies you need to fulfill before you start training your model

### Reffer to version chart for reffernce
  - https://www.tensorflow.org/install/source#gpu
  - In my case following versions are used:
    - python: 3.10.12
      - `python --version`
    - tensorflow: 2.15.0
      - `python3 -c "import tensorflow as tf; print(tf.__version__)"`
    - CUDA: 12.2
      - `nvcc --version`
    - cuDNN: 8.9

### Setup env and install requirements
  - Note:
    - These steps are for Linus based system (WSL2 in my case)
    - For Windows tenforflow 2.10 was the last version to support GPU. Post that you need to install tensorflow in WSL  
    - In case GPU are not abailable, CPU will be used. Although it will be 3 times slower
  - `conda create -n wsl_tf_p31012 python=3.10.12`
  - Common libs: `conda install numpy pandas seaborn xgboost matplotlib tqdm pillow`
  - Tensorflow: `python -m pip install tensorflow[and-cuda]`
    - This will install tensorflow supporting parallel computing on GPU using CUDA
  - Check GPU visibility: `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`
  - Also add cuda to PATH: `export PATH=/usr/local/cuda/bin/:$PATH`
  - Test if CUDA is correctly installed: `nvcc --version`


### Install CUDA for parallel computing on GPU
  - Go to https://developer.nvidia.com/cuda-toolkit-archive and choose apt version
  - In our case it will take us to https://developer.nvidia.com/cuda-12-2-2-download-archive as we want to install 12.2.x
    - Choose apt Arch and ditro (and version)
    - Choose network for installation
    - Follow the instructions provided after that you will install cudnn-cuda-12
    - In My case following commands were executed:
      ```bash
      wget https://developer.download.nvidia.com/compute/cuda/12.2.2/local_installers/cuda_12.2.2_535.104.05_linux.run
      sudo sh cuda_12.2.2_535.104.05_linux.run
      ```


### Install CUDA Deep Neural Network (cuDNN) for Deep learning neaural network computing
  - Go to https://developer.nvidia.com/rdp/cudnn-archive and choose apt version (might need login)
  - In our case we chose 8.9.7 version
  - Download the deb file and install using
    - `sudo dpkg -i <absolute path>`
  - Test if cuDNN is correctly installed:
    ```bash
    sudo apt-get install libcudnn9-samples
    cd /usr/src/cudnn_samples_v9/mnistCUDNN
    sudo make clean
    sudo make
      # If case you see and error like this: 'test.c:1:10: fatal error: FreeImage.h: No such file or directory' run following
      sudo apt-get install libfreeimage3 libfreeimage-dev
    ./mnistCUDNN
      #It will output 'Test Passed!' in the end, which means cudnn is installed correctly
     ```


## Notes

* Due to the limitations in my harware The model is trained with 20000 tweets with equal numbers of positive and negative tweets.
