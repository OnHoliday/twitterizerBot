from utils_ import *
import os
from tempfile import gettempdir
from gensim.models import Word2Vec
import pandas as pd
from nltk.tokenize import RegexpTokenizer

import  logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#articles = []    
#with jsonlines.open(r'sample-1M.jsonl') as reader:
#    for obj in reader:
#            articles.append(obj)
## Transform the list to a dataframe
#df = pd.DataFrame(articles)
#df = df[['title','content']]
#
#list_of_indexes = []
#part_size = 10000
#for i in range(0,len(df), part_size):
#    list_of_indexes.append((i, i+part_size))
#
#for left, right in list_of_indexes:
#    index = int(left/part_size)
#    globals()['df_temp_%s' % index] = df[left:right]
#    
#for index in range(int(left/part_size)+1):
#    globals()['df_temp_%s' % index].to_csv(r'C:\Users\Konrad\Desktop\NOVA IMS\text mining\Project\df_part_' + str(index) + '.csv',index=False)


## initialize model -> do this for a first time after that load model
#model = gensim.models.Word2Vec(input_to_wordVec, min_count=1)

# train model more
model = gensim.models.Word2Vec.load('mymodel')
#index= 4
for index in range(4, 100, 1): 
    print(index)
    temp_df = pd.read_csv(r'C:\Users\Konrad\Desktop\NOVA IMS\text mining\Project\df_part_' + str(index) + '.csv', header = 0, sep=',')

    tokenizer = RegexpTokenizer(r'\w+')

    input_to_wordVec = []
    for i, value in temp_df.iterrows():
        input_to_wordVec.append(tokenizer.tokenize(value['content']))
    
    model.train(input_to_wordVec, total_examples=10000, epochs=1)
    accuracy = model.accuracy(r'C:\Users\Konrad\Desktop\NOVA IMS\text mining\evaluete_accuracy_dataset.txt')
    model.save('mymodel')


