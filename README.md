# CIL_Project
## What to do before running
* Download Twitter Dataset from Kaggle Competition Page and put in a folder named twitter-datasets in the folder CIL.
* Download from https://github.com/google-research/bert either the bert_base Model or the bert_large_wwm Model (Whole Word Masking) and put it in the folder CIL/bert/checkpoints/bert_base or CIL/bert/checkpoints/bert_large_wwm.
* To run everything now on Google Colab, upload everthing to your Google Drive and then run the Notebook on TPUs if available.
## Other resources/papers
* [Trainig data set](https://www.kaggle.com/kazanova/sentiment140)
* [Google's T5](https://arxiv.org/pdf/1910.10683.pdf) instead of BERT? 
  * [Announcement with Colab Link](https://ai.googleblog.com/2020/02/exploring-transfer-learning-with-t5.html)
* [Albert](https://ai.googleblog.com/2019/12/albert-lite-bert-for-self-supervised.html) instead of BERT
* [GPT-3](https://arxiv.org/abs/2005.14165) is not much different to GPT-2 for much training and not available to public.
* [Preprocessing](https://trec.nist.gov/pubs/trec28/papers/DICE_UPB.IS.pdf)
  * Stop-words, URLs, usernames and unicode-characters are removed
  * Extra white-spaces, repeated full stops, question marks and exclamation marks are removed.
  * Emojis were converted to text using the python libraryemoji4
  * Lemmatization,  restoring  language  vocabulary  to  general  form  (can  express  completesemantics) by WordNetLemmatizer5.
  * Finally all tweet tokens are converted to lower-case.
* [Sample project on COVID](https://arxiv.org/pdf/2005.07503.pdf)

  
