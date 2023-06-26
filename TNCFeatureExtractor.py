import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
from string import punctuation
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import re
from sklearn.feature_extraction.text import TfidfVectorizer


sent_stopwords=stopwords.words('english')
punctuation+='\n'

def remove_numeric_statements(text):
      # Remove numeric statements
      text = re.sub(r'\b\d+\b', '', text)

      # Remove punctuations
      text = re.sub(r'[^\w\s]', '', text)

      return text

def reduce_to_root_words(sentence):
    stemmer = PorterStemmer()
    words = word_tokenize(sentence)
    root_words = [stemmer.stem(word) for word in words]
    reduced_sentence = ' '.join(root_words)
    return reduced_sentence

class TNC_Feature_Extractor:
    def __init__(self,text):
        self.text=text
        self.sent_stopwords=stopwords.words('english')
        self.punctuation=punctuation

    def tokenize_to_sents(self):
      tnc=self.text
      sentences=tnc.split(".")
      sentences=" ".join(sentences)
      sentences=sentences.split("\n")
      non_empty_sents=[]
      for sent in sentences:
        if(len(sent) > 2):
            non_empty_sents.append(sent)

      return non_empty_sents

    
    def preprocess_text(self,sentences):
      # remove punct and alpha num
      preprocessed_sents=[remove_numeric_statements(x) for x in sentences]
      # stemming
      preprocessed_sents=[reduce_to_root_words(x) for x in preprocessed_sents]

      return preprocessed_sents
      

class TFIDFscorer:
  def __init__(self):
    self.vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'))
  
  def generate_words_to_scores(self,preprocessed_sents):
    tfidf_matrix = self.vectorizer.fit_transform(preprocessed_sents)
    all_words=self.vectorizer.get_feature_names_out()
    word_to_score={}
    for word in all_words:
        indx= self.vectorizer.vocabulary_.get(word)
        word_to_score[word]=self.vectorizer.idf_[indx]
    #     print("{}: Score-> {}".format(word,v.idf_[indx]))
    return word_to_score