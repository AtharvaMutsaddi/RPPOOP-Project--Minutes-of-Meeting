from flask import Flask, request, jsonify, render_template, url_for
import numpy as np
# CUSTOM CLASSES:
from TranscriptSummarizerObj import YTpreprocessor,SummaryGenerator
from TNCFeatureExtractor import TNC_Feature_Extractor,TFIDFscorer


import warnings
warnings.filterwarnings("ignore")
# import pickle
from email.message import EmailMessage
import ssl
import smtplib 

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
import matplotlib.pyplot as plt
from flask import Flask, render_template, request
from wordcloud import WordCloud
from collections import Counter
import os

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
def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('static/wordcloud.png')
    plt.clf()
def generate_bar_chart(words, counts):
    plt.figure(figsize=(10, 5))
    plt.bar(words, counts)
    plt.xlabel('Words',fontsize=16)
    plt.ylabel('Frequencies',fontsize=16)
    plt.title('Top 10 Word Frequencies',fontsize=16)
    plt.xticks(rotation=45)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig('static/barchart.png')
    plt.clf()
def generate_pie_chart(keywords,size):
    plt.figure(figsize=(8, 8))
    word_counts = Counter(keywords)
    # Calculate the percentage of each word in the text
    percentages = [(word, count / size * 100) for word, count in word_counts.items()]

    # Sort the percentages in descending order
    percentages = sorted(percentages, key=lambda x: x[1], reverse=True)

    # Select the top 10 words and their percentages
    top_words = [word for word, _ in percentages[:10]]
    top_percentages = [percentage for _, percentage in percentages[:10]]

    # Plotting the pie chart
    plt.pie(top_percentages, labels=top_words, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 15})

    # Aspect ratio of the pie chart to make it a circle
    plt.axis('equal')

    # Saving the pie chart as an image
    plt.savefig('static/pie_chart.png')

    # Clear the current figure
    plt.clf()
@app.route('/textanalyzer')
def textanalyzer():
    return render_template('textanalyzer.html')
    
@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.form['text']
    size=len(text.split())
    # Delete existing image files
    existing_files = ['static/wordcloud.png', 'static/barchart.png','static/pie_chart.png']
    for file in existing_files:
        if os.path.exists(file):
            os.remove(file)

    # Generate new images
    text=remove_numeric_statements(text)
    keywords=[]
    for word in text.split():
        if word not in stopwords.words('english'):
            keywords.append(word)
    important_txt=" ".join(keywords)
    word_counts = Counter(keywords)
    top_words = [word for word, count in word_counts.most_common(10)]
    top_counts = [count for word, count in word_counts.most_common(10)]
    generate_wordcloud(important_txt)
    generate_bar_chart(top_words, top_counts)
    generate_pie_chart(keywords, size)
    return render_template('result.html')

if __name__=="__main__":
    app.run(debug=True)
