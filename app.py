from flask import Flask, request, jsonify, render_template, url_for
import numpy as np
from youtube_transcript_api import YouTubeTranscriptApi as yta
import pickle




def segment_text(text, block_length):
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

def loadmodel():
    print("Loading Model In backend...")
    summarizer=pickle.load(open("model_pickle","rb"))
    print("Successfully Loaded!\n")
    return summarizer


app = Flask(__name__)


@app.route('/')
@app.route('/home')
def home():
    return render_template('index.html')


@app.route('/summarize', methods=['POST'])
def summarize():
    summarizer=loadmodel()
    id = [str(x) for x in request.form.values()]
    ytlink = id[0]
    ytlinkid = ytlink.split("v=")[-1]
    data = yta.get_transcript(ytlinkid)
    final_tra = " ".join([x['text'] for x in data])
    block_length = 500  # Set the desired length of each text block
    segments = segment_text(final_tra, block_length)
    final_summary=""
    n=len(segments)
    i=1
    word_count_original=len(final_tra.split())
    
    for text in segments:
        print(f"Summarizing {i} out of {n} segments of the transcript")
        max_len=min(150,len(text))
        summary=summarizer(text,max_length=max_len,min_length=30,do_sample=False)
        final_summary+=" "
        final_summary+=summary[0]['summary_text']
        i+=1
    word_count_final=len(final_summary.split())
    return render_template('index.html', final_tra=final_tra,final_summary=final_summary,word_count_original=word_count_original,word_count_final=word_count_final)






if __name__=="__main__":
    app.run(debug=True)