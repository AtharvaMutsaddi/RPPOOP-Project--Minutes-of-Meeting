from youtube_transcript_api import YouTubeTranscriptApi as yta
from transformers import pipeline
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
