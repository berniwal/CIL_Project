import pandas as pd
import csv

def load():
    df = pd.read_csv('training.1600000.processed.noemoticon.csv', encoding='latin-1')
    df.columns = ['target', 'ids', 'date', 'flag', 'user', 'text']
    return df

def load_pos():
    df=pd.read_csv('pos.txt', encoding='latin-1')
    df.columns = ['text']
    return df

def load_neg():
    df=pd.read_csv('neg.txt', encoding='latin-1')
    df.columns = ['text']
    return df

def write_pos(df):
    pos = df[df['target']==4]
    pos = pos['text']
    pos.to_csv('pos.txt', header=False, index=False)

def write_neg(df):
    neg = df[df['target']==0]
    neg = neg['text']
    neg.to_csv('neg.txt', header=False, index=False)

def write_pre_pos(df):
    df.to_csv('preprocessed_pos.txt', header=False, index=False, quoting=csv.QUOTE_NONE, escapechar='\t')

def write_pre_neg(df):
    df.to_csv('preprocessed_neg.txt', header=False, index=False, quoting=csv.QUOTE_NONE, escapechar='\t')

def remove_user(df):
    '''
    replace @something with <user>
    '''
    for index, row in df.iterrows():
        string = row['text']
        for word in string.split(' '):
            if len(word)>0 and word[0]=='@':
                string = string.replace(word, '<user>')
        df.at[index,'text'] = string
    return df

def remove_url(df):
    '''
    replace http:// or https:// or www. with <url>
    '''
    for index, row in df.iterrows():
        string = row['text']
        for word in string.split(' '):
            if len(word)>0 and (word.startswith('http://') or word.startswith('https://') or word.startswith('www.')):
                string = string.replace(word, '<url>')
        df.at[index,'text'] = string
    return df
        



if __name__=='__main__':
    df = load()
    write_pos(df)
    write_neg(df)
    pos = load_pos()
    neg = load_neg()
    pos = remove_user(pos)
    neg = remove_user(neg)
    pos = remove_url(pos)
    neg = remove_url(neg)
    write_pre_pos(pos)
    write_pre_neg(neg)