from flask import Flask, request, jsonify, render_template, url_for
import numpy as np
from youtube_transcript_api import YouTubeTranscriptApi as yta
import warnings
warnings.filterwarnings("ignore")
# import pickle
from email.message import EmailMessage
import ssl
import smtplib 
from transformers import pipeline
# TNC SUMMARIZER MODEL
import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
from string import punctuation
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
import re
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def send_email(email_sender,email_password,email_reciever,summary):
    subject="Your Meeting Summary"
    username=email_reciever.split("@")[0]
    body=""""""
    body+=f"Hello {username}. Here's the summary of the video/meeting you requested:-"
    body+=summary
    em= EmailMessage()
    em['From']=email_sender
    em['To']= email_reciever
    em['Subject']=subject
    em.set_content(body)
    context=ssl.create_default_context()
    with smtplib.SMTP_SSL('smtp.gmail.com',465,context=context) as smtp:
        smtp.login(email_sender,email_password)
        smtp.sendmail(email_sender,email_reciever,em.as_string())
        print("Email sent successfully")

class YTpreprocessor:
    def __init__(self,ytlinkID):
        self.ytlinkID=ytlinkID
        self.tra=""
    def generate_preprocessed_transcript(self):
        data = yta.get_transcript(self.ytlinkID)
        self.tra=" ".join([x['text'] for x in data])
        return self.tra
    
    def segment_text(self,text, block_length=500):
        segments = []
        current_segment = ""
        words = text.split()

        for word in words:
            if len(current_segment.split()) + 1 + 1 <= block_length:  # Add 1 for the space between words
                current_segment += word + " "
            else:
                segments.append(current_segment)
                current_segment = word + " "

        # Add the last segment
        segments.append(current_segment)
        return segments

class AbstractSummarizer:
    def loadmodel(self):
        summarizer=pipeline('summarization')
        return summarizer

class SummaryGenerator(AbstractSummarizer):
    def __init__(self):
        self.summarizer=self.loadmodel()

    def get_timeframe_wise_summary(self,segments):
        final_summary=""
        n=len(segments)
        i=1
        for text in segments:
            print(f"Summarizing {i} out of {n} segments of the transcript")
            max_len=min(150,len(text))
            summary=self.summarizer(text,max_length=max_len,min_length=30,do_sample=False)
            final_summary+=" "
            final_summary+=summary[0]['summary_text']
            i+=1
        
        return final_summary


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
    
def calculate_sentence_scores(preprocessed_sents,non_empty_sents,word_to_score):
    sent_score_dict={}
    for i in range(len(preprocessed_sents)):
        sent=preprocessed_sents[i]
        score=0
        n=len(sent.split(" "))
        sent_array=sent.split()
        for word in sent_array:
            if word in word_to_score.keys():
                score+=word_to_score[word]
        if(n>10):
            score=float(score/n)
        else:
            score=0
        
        sent_score_dict[non_empty_sents[i]]=score
            
    
    return sent_score_dict

def get_top_sentences(sentence_scores, percentage):
    sorted_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)
    num_sentences = len(sorted_sentences)
    top_sentences_count = int(num_sentences * percentage)
    top_sentences = dict(sorted_sentences[:top_sentences_count])
    return top_sentences
def calculate_cosine_similarity(reference_text, summary):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([reference_text, summary])
    similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
    return similarity*100
# ####################################


app = Flask(__name__)


@app.route('/')
@app.route('/home')
def home():
    return render_template('index.html')


@app.route('/summarize', methods=['POST'])
def summarize():
    # LOAD MODEL
    sg=SummaryGenerator()
    # EXTRACT ID FROM FORM INPUT
    id = [str(x) for x in request.form.values()]
    ytlink = id[0]
    ytlinkid = ytlink.split("v=")[-1]
    # INITIALIZE YT PREPROCESSOR
    ytp=YTpreprocessor(ytlinkid)
    # GET ORIGINAL TRANSCRIPT
    final_tra=ytp.generate_preprocessed_transcript()
    word_count_original=len(final_tra.split())

    # SEGMENTING TRANSCRIPT WITH RESPECT TO WORD BLOCKS
    block_length = 500  # Set the desired length of each text block
    segments = ytp.segment_text(final_tra,block_length)

    # GENERATING A COMPILED SUMMARY
    final_summary=sg.get_timeframe_wise_summary(segments)
    word_count_final=len(final_summary.split())

    send_email("mspaakhi01@gmail.com","xjmftwwqkywqlydc","atharva.mutsaddi@gmail.com",final_summary)
    return render_template('index.html', final_tra=final_tra,final_summary=final_summary,word_count_original=word_count_original,word_count_final=word_count_final)

@app.route('/upload')
def upload():
    return render_template('tnc.html')
    
@app.route('/summarizetnc', methods=['POST'])
def summarizetnc():
    file = request.files['file']
    tnc = file.read().decode('utf-8')
    extractor=TNC_Feature_Extractor(tnc)
    sentences=extractor.tokenize_to_sents()
    preprocessed_sents=extractor.preprocess_text(sentences)
    scorer=TFIDFscorer()
    word_to_score=scorer.generate_words_to_scores(preprocessed_sents)
    sent_scores=calculate_sentence_scores(preprocessed_sents,sentences,word_to_score)
    top_sents_dict=get_top_sentences(sent_scores,0.2)
    final_sents=[]
    final_para=""
    for sent in top_sents_dict.keys():
        final_para+=sent
        final_sents.append(sent)
    # print(final_para)
    score=calculate_cosine_similarity(final_para,tnc)
    return render_template('tnc.html', final_sents=final_sents, score=score)
if __name__=="__main__":
    app.run(debug=True)