import os
import numpy as np

from train import create_model
from bert.tokenization.bert_tokenization import FullTokenizer

CHECKPOINT = './bert/checkpoints/bert_base'
CHECKPOINT_VOCAB = os.path.join(CHECKPOINT, 'vocab.txt')

model = create_model(23, adapter_size=None)
model.load_weights("twitter_initial.h5")

pred_sentences = [
  "<user> yay ! ! #lifecompleted . tweet / facebook me to let me",
  "workin hard or hardly workin rt <user> at hardee's with my future",
  "<user> check my tweet pic out . that was the outfit before . this is",
  "wish i could be out all night tonight ! <user>",
  "<user> i got kicked out the wgm",
  "rt <user> <user> <user> yes she is ! u tell it ! my lips are"
]

tokenizer = FullTokenizer(vocab_file=CHECKPOINT_VOCAB)
pred_tokens = map(tokenizer.tokenize, pred_sentences)
pred_tokens = map(lambda tok: ["[CLS]"] + tok + ["[SEP]"], pred_tokens)
pred_token_ids = list(map(tokenizer.convert_tokens_to_ids, pred_tokens))

pred_token_ids = map(lambda tids: tids +[0]*(23-len(tids)),pred_token_ids)
pred_token_ids = np.array(list(pred_token_ids))

print('pred_token_ids', pred_token_ids.shape)

res = model.predict(pred_token_ids).argmax(axis=-1)

for text, sentiment in zip(pred_sentences, res):
  print(" text:", text)
  print("  res:", ["negative","positive"][sentiment])

