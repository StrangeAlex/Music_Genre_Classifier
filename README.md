#  Music_Genre_Classifier
My HSE ML course project >:)
Presentation link [here](https://docs.google.com/presentation/d/1ORDRERDKg60GDq_dkvli3XjH_5LoEhLaHTteGsv0TT0/edit?usp=sharing)!

### Installing dependencies
```console
pip install -r requirements.txt
```

##  Source data description
For model training two datasets were used:
- [EDM Music Dataset](https://www.kaggle.com/datasets/sivadithiyan/edm-music-genres/data) from Kaggle.
- Parsed YouTube music containing about 100 songs for each of 3 genres: <b>pop, rock and classical</b>. More info on that in [Notebooks/dataset_creation.ipynb](Notebooks%2Fdataset_creation.ipynb).

Each of 19 classes (genres) is represented equally, resulting in 2000 train samples and 500 test samples.

In both datasets splitting is done AFTER feature extraction, so no data leakage here. 

##  Preprocessing description
First of all, audio is loaded into Python library <b>librosa</b>.

Then it is sliced into 3 seconds long audio clips.

Finally, feature extraction is done on each clip, retrieving 130 useful audio characteristics such as <b>spectral centroid and MFCCs</b>. 

It's that simple!

## Used models and evaluation metrics
A total of 5 models were tested:
- Logistic Regression
- Random Forest
- SVM
- Catboost
- Naive Bayes Classificator

In the end, Catboost and SVM (with rbf kernel) were superior, each of them scoring around 0.73 on F1. This may seem low, but if you look at confusion matrix, you may see that mistaken classes are somewhat familiar (e.g. rock music is often considered pop).

More info on model evaluations can be found in [Notebooks/models_comparison.ipynb](Notebooks%2Fmodels_comparison.ipynb).

## Making prediction on real data
To predict song genre I do the following:
1. Split input audio into 3 second clips
2. Predict genre using SVM on each of them
3. Count up each prediction
4. Display occurrence of each genre in %.

This way audio length proportions are preserved during feature extraction and the result contains different genres (which makes sense, since it's often hard to predict just one genre).

All these steps are implemented in [Telegram_bot/bot.py](Telegram_bot%2Fbot.py).

Bot link [here](https://t.me/music_genre_cls_bot).
