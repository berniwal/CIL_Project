# CIL_Project
## What to do before running CIL-Notebook.ipynb
* [Tutorial](https://github.com/kpe/bert-for-tf2)
* Download Twitter Dataset from [Kaggle Competition Page](https://www.kaggle.com/c/cil-text-classification-2020) and put in a folder named twitter-datasets in the folder CIL. (only neg-full.txt needed, because too big for GitHub)
* Read and do things in README of CIL/bert/checkpoints/bert_base or CIL/bert/checkpoints/bert_large_wwm.
* To run everything now on Google Colab, upload everthing to your Google Drive and then run the Notebook on TPUs if available.

## What to do before running ALBERT2.ipynb
* Download [ALBERT](https://github.com/google-research/ALBERT) pretrained model and put it in './bert/checkpoints/albert_large_v2'.
* From the library that we use, go to CIL_Project/CIL/bert/tokenization/albert_tokenization.py and uncomment #import sentencepiece as spm.
* From there, the script makes the same steps as the original Notebook and the only differences are the creating of the model loads ALBERT weights and has shared_layer and embedding_size for ALBERT according to [library readme](https://github.com/kpe/bert-for-tf2)
* The other difference is the preprocessing that uses the ALBERT tokenizer and the sentencepiece model, but they have the same interface/class as the original tutorial(FullTokenizer).
* Other changes are only directories.
* Current Learning on Large

## Other resources/papers
### Other models
* [Google's T5](https://arxiv.org/pdf/1910.10683.pdf) instead of BERT?
  * [Announcement with Colab Link](https://ai.googleblog.com/2020/02/exploring-transfer-learning-with-t5.html)
  * [Huggingface implementation](https://huggingface.co/transformers/model_doc/t5.html)
* [Albert](https://ai.googleblog.com/2019/12/albert-lite-bert-for-self-supervised.html) instead of BERT
* [GPT-3](https://arxiv.org/abs/2005.14165) is not much different to GPT-2 for much training and not available to public.
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
* MoNoise [Model](https://www.aclweb.org/anthology/P19-3032.pdf) [Rules](https://arxiv.org/pdf/1710.03476.pdf)
  * For  each  word  we  find  the  top  40  closest  candidates  in  thevector space based on the cosine distance
  * We use the [Aspell](http://aspell.net/) spell checker to repair typographical errors
  * We generate a list of all replacement pairs occurring in the training data.
  * We include a generation module thatsimply searches for all words in the Aspell dictionary which start with the character sequence of ouroriginal word.  To avoid large candidate lists.
  * We generate word splits by splitting a word on every possible position and checking if bothresulting words are canonical according to the Aspell dictionary.
* [MoNoise alternative](https://arxiv.org/pdf/1904.06100.pdf)
* [Lexical Normalization with BERT](https://www.aclweb.org/anthology/D19-5539.pdf)
* [Twitter Fine-Tuning BERT](https://arxiv.org/pdf/1905.05583.pdf)
### Adverserial
* [Adverserial NLP Overview](https://www.aclweb.org/anthology/N19-5001/)
* [Adverserial with RNNs](https://www.aclweb.org/anthology/L18-1584.pdf)
### Links to Google Drive Folders
* [Bert Base with MoNoise without badspeller and without extra data (Manuel) and Albert Base without MoNoise and without extra data](https://drive.google.com/drive/folders/1ynCZnjcYXVg_qZtam3bAbXrEUiI4CyqI?usp=sharing)
* [AlBert large, no Monoise, no additional data](https://drive.google.com/drive/folders/1bon0OFwJRQRY1rsEiWRKOhNugxcZ5sPg?usp=sharing)
