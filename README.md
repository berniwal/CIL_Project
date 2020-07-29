# CIL_Project
This Repository contains code from [bert-for-tf2](https://github.com/kpe/bert-for-tf2) and CIL-Notebook builds their provided [Example](https://github.com/kpe/bert-for-tf2/blob/master/examples/tpu_movie_reviews.ipynb) for movie sentiment analysis.
## How to run CIL-Notebook.ipynb
* Clone the project from this GitHub repository.
* Download Twitter Datasets from [Dropbox](https://www.dropbox.com/sh/gvzo0jrnfhcnkeh/AACYlqypVkBYzhL_hyjWXRwNa?dl=0) and put them in the folder `CIL/twitter-datasets`. 
* Download corresponding checkpoints of the models you want to run (follow readme in specific `CIL/bert/checkpoints/"model_name"` folder), checkpoint for the different Models need to be put into the `CIL/bert/checkpoints/"model_name"` folder. For example for our best performing solution download https://storage.googleapis.com/albert_models/albert_xxlarge_zh.tar.gz and unpack it in the `CIL/bert/checkpoints/albert_xxlarge` folder.
* To run everything now on Google Colab, upload everthing to your Google Drive and then run the Notebook on TPUs if available. 
* Cell 3 contains the config for the notebook: 
  * set `MODEL= 'bert'` or `'bert_large'` or `'albert'` to change the model
  * set `ADDITIONAL_DATA = True` or `False` to toggle the use of extra data. 
  * set `DATASET_PREPROCESSING = ' '` or `'_monoise'` or `'_monoise_b'` to select your pre-processing
* Note for albert you have to change the `CHECKPOINT` parameter to the corresponding albert version you want to run! Currently this is set to `'./bert/checkpoints/albert_xxlarge'`, for training a albert_xlarge model, just change the path to `'./bert/checkpoints/albert_xlarge'` and make sure you have put the checkpoints in the corresponding folder.
* ***IMPORTANT*** When training the largest models like bert_large, albert_xxlarge, albert_xlarge it is possible to run into memory issues, in this case you can not run them unless you get assigned a high ram runtime from google or have one as you have a google colab pro subscription. 
* For bert_xlarge and all the albert models training we used `CURRENT_LEARNING_RATE=1e-6`, for all the others we set `CURRENT_LEARNING_RATE=1e-5`.
## Reproduce results of our best solution
### Retraining
* Set `MODEL='albert'`, `ADDITIONAL_DATA=True`, `ADDITIONAL_DATA_ONLY_FIRST=True` and `DATASET_PREPROCESSING='_monoise'`
* Make sure to have put the albert_xxlarge checkpoint in the `'./bert/checkpoints/albert_xxlarge'` folder and that `CHECKPOINT='./bert/checkpoints/albert_xxlarge'` for the `"if MODEL=='albert':'` case.
* Make sure to have downloaded the MoNoise training and testing data for the addititional data (`extra_pos_monoise.txt, extra_neg_monoise.txt`) and the initial data (`train_pos_full_monoise.txt, test_pos_full_monoise.txt, train_neg_full_monoise.txt, train_neg_full_monoise.txt`) from [Dropbox](https://www.dropbox.com/sh/gvzo0jrnfhcnkeh/AACYlqypVkBYzhL_hyjWXRwNa?dl=0) and put in the `'CIL/twitter-datasets'` folder.
* Now you can run the Notebook from top to bottom. (albert_xxlarge training can take a while, one epoch takes about half a day and the first epoch with the additional data about one day)
* ***TIPP*** To prevent Google Colab from disconnecting after 4 hours use: https://stackoverflow.com/questions/57113226/how-to-prevent-google-colab-from-disconnecting/59026251#59026251
### Model reloading and generate prediction.
* If you don't want to retrain the model you can also just generate our final prediction by downloading our best checkpoint from [Dropbox](https://www.dropbox.com/sh/gvzo0jrnfhcnkeh/AACYlqypVkBYzhL_hyjWXRwNa?dl=0).
* Make sure to have set the parameters in the first cell according to the first and second step in the Retraining section, and make sure to have the testing data mentioned there correctly uploaded. Then run all cells up to and including the cell below the Build Model Text.
* Also upload the checkpoint to Google Drive and set the `use_checkpoint` parameter in the third lowest cell to the directory you have put the checkpoint + checkpoint_file_name, in our case this is just `'albert_xxlarge_additional_monoise_h5'`.
* Now run the last 4 cells to generate the prediction and the corresponding csv file.
* **IMPORTANT** If you run this you might run into out of memory issues, as our largest model has a lot of parameters. In this case you have either to hope that google will assign to you a high ram runtime or you need to have a google colab pro subscription and run it there with a high ram runtime.
### How to generate the learning plots
* Make sure to have the folder `results/results_final` with all the `*_final.txt` files uploaded to google drive, if you have cloned the repo and uploaded it with everything you are fine.
* Run the first two cells to initialize your runtime for the notebook.
* Then go to the cell below the 'Generate Plots' cell
* Adjust the save_label to the plots you want to see, the options are:
  * `albert_bert` : generates a plot where all the bert models and albert models without lexical normalization and without monoise get plotted
  * `comparison_full`: generates a plot where all the bert and albert models with all the experiments for lexical normalization and monoise get plotted
  * `comparison_bert`: gererates a plot where all the bert models with all experiments for lexical normalization and monoise get plotted
  * `comparison_albert`: generates a plot where all the albert models with all experiments for lexical normalization and monoise get plotted
## How to generate MoNoise data
* If you want to generate the MoNoise data yourself just follow the steps below, otherwise you can also download the MoNoise processed data files from [Dropbox](https://www.dropbox.com/sh/gvzo0jrnfhcnkeh/AACYlqypVkBYzhL_hyjWXRwNa?dl=0).
* Clone [MoNoise repo](https://bitbucket.org/robvanderg/monoise/src/master/)
* Follow "Example run" instructions in repo readme to compile and test MoNoise with an example
* copy twitterdata folder to monoise folder
* in monoise/src run command ```./tmp/bin/binary -r ../data/en/chenli -m RU -C -b -t -d ../data/en -i ../twitterdata/train_pos_full.txt -o ../results/result_pos_full.txt" (-b is bad-speller mode)```
* to run on euler cluster in chunks: 
  * connect to [Euler](https://scicomp.ethz.ch/wiki/Getting_started_with_clusters)
  * use ```scp``` to copy monoise folder to Euler cluster
  * load gcc with ```module load new && module load gcc/6.3.0``` 
  * use ```split -l 200000 train_pos_full.txt train_pos_full --additional-suffix=.txt``` if you want to split data in chunks
  * use ```bsub -W 48:00 -R "rusage[mem=4096]" "./tmp/bin/binary -r ../data/en/chenli -m RU -C -b -t -d ../data/en -i ../twitterdata/train_pos_fullaa.txt -o ../results/result_pos_fullaa.txt"``` to submit job (-W 24:00 sufficient when not using bad-speller) (run for all chunks) 
  * use ```cat result_pos_fullaa.txt result_pos_fullab.txt result_pos_fullac.txt result_pos_fullad.txt result_pos_fullae.txt result_pos_fullaf.txt result_pos_fullag.txt > result_pos_full.txt``` to merge chunks

## Other files
Jannik UMAP
Manuel Länge
## Other resources/papers
### Models
* [ALBERT](https://ai.googleblog.com/2019/12/albert-lite-bert-for-self-supervised.html) instead of BERT
* [BERT explanation](http://jalammar.github.io/illustrated-bert/)
* [BERT tutorial](https://github.com/kpe/bert-for-tf2)
### Preprocessing
* [Trainig data set](https://www.kaggle.com/kazanova/sentiment140)
* [Preprocessing](https://trec.nist.gov/pubs/trec28/papers/DICE_UPB.IS.pdf)
  * Stop-words, URLs, usernames and unicode-characters are removed
  * Extra white-spaces, repeated full stops, question marks and exclamation marks are removed.
  * Emojis were converted to text using the python libraryemoji4
  * Lemmatization,  restoring  language  vocabulary  to  general  form  (can  express  completesemantics) by WordNetLemmatizer5.
  * Finally all tweet tokens are converted to lower-case.
* [Sample project on COVID](https://arxiv.org/pdf/2005.07503.pdf)
### Lexical normalization
* MoNoise [Model](https://www.aclweb.org/anthology/P19-3032.pdf) [Rules](https://arxiv.org/pdf/1710.03476.pdf) [Code](https://bitbucket.org/robvanderg/monoise/src/master/)
  * For  each  word  we  find  the  top  40  closest  candidates  in  thevector space based on the cosine distance
  * We use the [Aspell](http://aspell.net/) spell checker to repair typographical errors
  * We generate a list of all replacement pairs occurring in the training data.
  * We include a generation module thatsimply searches for all words in the Aspell dictionary which start with the character sequence of ouroriginal word.  To avoid large candidate lists.
  * We generate word splits by splitting a word on every possible position and checking if bothresulting words are canonical according to the Aspell dictionary.
* [Lexical Normalization with BERT](https://www.aclweb.org/anthology/D19-5539.pdf)
* [Twitter Fine-Tuning BERT](https://arxiv.org/pdf/1905.05583.pdf)
### Links to Google Drive Folders
* [Bert Base with MoNoise without badspeller and without extra data (Manuel) and Albert Base without MoNoise and without extra data](https://drive.google.com/drive/folders/1ynCZnjcYXVg_qZtam3bAbXrEUiI4CyqI?usp=sharing)
* [AlBert large, no Monoise, no additional data](https://drive.google.com/drive/folders/1bon0OFwJRQRY1rsEiWRKOhNugxcZ5sPg?usp=sharing)
