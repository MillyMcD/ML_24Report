# Spotify ML
This is a project that explores the top 10,000 songs played on Spotify

## Installation
To install this project, call
```
sh installation.sh
``` 
This will create an environment called `spotify_env`. do this only once, or only if you want to reinstall the project

## Spinning up the notebook server
Once installed, you can deploy the project by calling 

```
source run.sh
```

This will spin up a jupyter notebook server on port 9001.

# Sent2Vec
this project uses the sent2vec pretrained model for encoding model weights. Go to this repository:

```
https://github.com/epfml/sent2vec
```
and download the `twitter_unigrams.bin` model and rename it as `model.bin` here.