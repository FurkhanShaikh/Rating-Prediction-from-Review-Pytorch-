# Rating-Prediction-from-Review-Pytorch
This repository contains the Pytorch torchtext implementation using the dataset available at https://www.kaggle.com/jvanelteren/boardgamegeek-reviews.

The goal of this project is to predict a rating 0-10 given some text (review) as input. The Model used here is the Vanilla LSTM model with word vectors embeddings.

The dataset contains over 15m reviews and has lots of missing values and different characters. The datset is cleaned and only the review (comment) and ratings are kept. the reviews serve as the input to the network along with the embeddings. the out is a vector containing 11 values for each class.

## Requirements

The program is written in python 3. So if you use python 3.x, it should work fine
Other libraries like include
- Pandas
- Numpy
- Scikit
- pytorch
- torchtext

## To deploy/ run

Just install the required librabries. Use Anaconda if you prefer easy installation. 
To install pytorch, use the anaconda command conda 
 conda install -c pytorch torchtext 
 
After thats done, view the ipython file for the code and model description.

To deploy on you need pytorch>=1.6.0 and word_embeddings.pt file, Text.Field file and weghts file.

This three files should be created during the training itself. I cannot upload here because each file is larger than 150MB
I'lll try to upload on drive maybe.

Updated code
https://www.kaggle.com/furkhan194/rating-prediction-pytorch
 
## View the ipython file for the code explanation and working


