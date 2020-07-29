# CIL_Project
## What to do before running CIL-Notebook.ipynb
* Download Twitter Dataset from [Kaggle Competition Page](https://www.kaggle.com/c/cil-text-classification-2020) and put in a folder named twitter-datasets in the folder CIL. (only neg-full.txt needed, because too big for GitHub)
* Read and do things in README of CIL/bert/checkpoints/bert_base or CIL/bert/checkpoints/bert_large_wwm.
* To run everything now on Google Colab, upload everthing to your Google Drive and then run the Notebook on TPUs if available.
* Cell 3 contains the config for the notebook: 
  * set MODEL= 'bert' or 'bert_large' or 'albert' to change the model
  * set ADDITIONAL_DATA = True or False to toggle the use of extra data. 
  * set DATASET_PREPROCESSING = ' ' or '_monoise' or '_monoise_b' to select your pre-processing
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
