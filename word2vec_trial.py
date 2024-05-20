import gensim
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

#download necessary nltk data 
nltk.download('punkt')
nltk.download('stopwards')

#sample data 
text = "hi"

#preprocess the text 
stop_words = set(stopwords.words('english'))
tokens = word_tokenize(text.lower())
filtered_tokens = [word for word in tokens if word.isalpha() and word not in stop_words]

#prepare sentences for word2vec 
sentences = [filtered_tokens]

#train the word2vec model 
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

#save model for later use 
model.save("word2vec.model")

#to load model later...
# model = Word2Vec.load("word2vec.model")

#now test word2vec model 
#find similar words 
similar_words = model.wv.most_similar('sample')
print(similar_words)

#get the vector for a word 
word_vector = model.wv['sample']
print(word_vector)