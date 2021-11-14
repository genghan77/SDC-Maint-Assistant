############################################################################
#### Initializing sentence_transformers
############################################################################
print("Initializing sentence_transformers")

from sentence_transformers import SentenceTransformer, util
model_BertMini = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# model_Bert = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
# model_Glove = SentenceTransformer('sentence-transformers/average_word_embeddings_glove.6B.300d')

def sentence_embedding_sim(sentence_embedding1, sentence_embedding2):
    cosine_scores = util.pytorch_cos_sim(sentence_embedding1, sentence_embedding2).item()
    return cosine_scores

def sentence_sim(sentence1, sentence2, model):
    embedding1 = model.encode(sentence1)
    embedding2 = model.encode(sentence2)

    return sentence_embedding_sim(embedding1, embedding2)

print("Initializing sentence_transformers -- Done")


############################################################################
#### Adaptation of Bert Tokenization
############################################################################
print("Defining Bert Tokenization")

import string
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

import re
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

def preprocessing(text, debug=False):
    if debug: print(text)
    
    # Tokenize into Bert wordpiece - Text is converted to lower case
    tokens = tokenizer.tokenize(text)
    if debug: print(tokens)
    
    # Index the position of tokens
    index_tokens = {}
    i = 0
    for t in tokens:
        index_tokens[t] = i
        i += 1
    
    # Number removal
    tokens_nonum_index = [1 if len(re.sub(r'#?#?[0-9]+', '', t)) > 0 else 0 for t in tokens]    
    if debug: print(tokens_nonum_index)
    
    # Punctuation removal
    tokens_nopunc_index = [1 if t not in string.punctuation else 0 for t in tokens]
    if debug: print(tokens_nopunc_index)
    
    # Remove incomplete wordpiece tokens
    tokens_final_index = []
    i = 0
    while i < len(tokens):
        if tokens_nonum_index[i] == 1 and tokens_nopunc_index[i] == 1:
            if not tokens[i].startswith("##"):
                tokens_final_index.append(1)
            else:
                if tokens_final_index[i-1] == 1:
                    tokens_final_index.append(1)
                else:
                    tokens_final_index.append(0)
                    
        else:
            tokens_final_index.append(0)
        i += 1
        
    if debug: print(tokens_final_index)
    
    # Final remaining wordpiece tokens
    tokens_final = []
    i = 0
    while i < len(tokens):
        if tokens_final_index[i] == 1:
            tokens_final.append(tokens[i])
        i += 1
    if debug: print(tokens_final)
    
    # Recreate the sentences
    sentence_final = tokenizer.convert_tokens_to_string(tokens_final)
    
    return sentence_final

print("Defining Bert Tokenization -- Done")

############################################################################
#### Reading in dataset and generating embeddings upfront for the corpus
############################################################################
import pandas as pd
import pickle5 as pickle

# Read in the dataset
fields = ['WONUM','TASKDESC', 'TASKDESC - CATEGORY', 'WOC.TEXTS']
data = pd.read_csv('PLP - FINAL DATASET.csv', on_bad_lines='warn', usecols=fields)

# Fill nan with ''
data.fillna('', inplace=True)
    
def create_embeddings():
    print("Creating embeddings")

    data['TASKDESC_PROCESSED'] = data['TASKDESC'].map(lambda a: preprocessing(a))
    sentences = data['TASKDESC_PROCESSED']
    embeddings = model_BertMini.encode(sentences)

    #Store sentences & embeddings on disc
    with open('embeddings.pkl', "wb") as fOut:
        pickle.dump({'sentences': sentences, 'embeddings': embeddings}, fOut, protocol=pickle.HIGHEST_PROTOCOL)

    print("Creating embeddings -- Done")

def load_embeddings():
    print("Loading embeddings")
    
    with open('embeddings.pkl', "rb") as fIn:
        stored_data = pickle.load(fIn)
        stored_sentences = stored_data['sentences']
        stored_embeddings = stored_data['embeddings']

    data['EMBEDDING'] = stored_embeddings.tolist()

    print("Loading embeddings -- Done")

############################################################################
#### Find sentences from corpus that is most similar to the query sentence¶
############################################################################
print("Finding similar sentence")

def find_topK_sentences(query_sentence, k):
    query_sentence_processed = preprocessing(query_sentence)
    query_sentence_embedding = model_BertMini.encode(query_sentence_processed)
    data['SIM_SCORE'] = data['EMBEDDING'].map(lambda a: sentence_embedding_sim(a, query_sentence_embedding))
    return data.drop_duplicates(subset=['TASKDESC']).sort_values('SIM_SCORE',  ascending=False).head(k)