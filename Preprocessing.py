import os                                                     # TO READ WIKIARTICLES.
import json                                                   # TO READ FROM .json FILE.
import math                                                   # TO CALCULATE nDCG
import pandas as pd                                           # TO READ COURSE DATA.
import numpy as np                                            # TO SAVE DICTIONARY.  
from nltk.tokenize import word_tokenize                       # TO CLEAN DATA.
from nltk.corpus import stopwords                             # TO CLEAN DATA.
from sklearn.feature_extraction.text import TfidfVectorizer   # TO EXTRACT PHRASES.
import scipy.sparse                                           # TO SAVE MODEL.
from sklearn.metrics.pairwise import cosine_similarity        # FOR COSINE SIMILARITY MEASURE
from nltk.stem import PorterStemmer
import nltk
nltk.download('stopwords')
nltk.download('punkt')



class Preprocessing:
    
    def __init__(self, phrases, abbreviations):
        self.stop_words = list(set(stopwords.words('english')))
        self.common_word = ['introduction', 'overview', 'basic', 'various', 'review', 'course', 'student', 'content', 'academic', 'necessary', 'different']
        self.stop_words = self.stop_words + self.common_word
        self.phrases = phrases
        self.abbreviations = abbreviations

        
    def cleanData(self, text):
        """
            Input   : text to be cleaned.
            Returns : cleaned text string.
        """
        text = text.replace("'"," ").replace("\""," ").replace(";"," ").replace(","," ").replace("-"," ").replace(":"," ").replace("["," ").replace("]"," ").replace("."," ").replace("/"," ").lower()
        text = word_tokenize(text)
        text = [token for token in text if (token.isalnum() and not token.isnumeric() and token not in self.stop_words and len(token)>1) ]
        return " ".join(text)
    
    
    def phrase_replacement(self, intext):
        outtext = intext
        for phrase in self.phrases:
            phrase_ = phrase.replace("_", " ")
            if phrase_ in outtext:
                outtext = outtext.replace(" "+phrase_+" ", " "+phrase+" ")
        return outtext
    
    
    def abbreviation_replacement(self, intext):
        outtext = intext
        for abbreviation in self.abbreviations:
            if abbreviation in outtext:
                outtext = outtext.replace(" "+abbreviation+" ", " "+abbreviation+" "+self.abbreviations[abbreviation]+" ")
            elif self.abbreviations[abbreviation] in outtext:
                outtext = outtext.replace(" "+self.abbreviations[abbreviation]+" ", " "+abbreviation+" "+self.abbreviations[abbreviation]+" ")
        return outtext
    

    def readWikipediaData(self, base_directory):
        """
            Input   : directory path.
            Returns : Wikipedia article titles and contents ordered list.
        """
        files = os.listdir(base_directory)
        content = {}

        for file in files:
            file_path = base_directory + file
            data = json.load(open(file_path,"r",encoding="utf-8"))
            for title in data['pages']:
                content[title] = self.cleanData(data['pages'][title]['text'] + title)
        return content


    def readCourseData(self, document_filename):
        """
            Input   : File path.
            Returns : Course titles and Course details ordered list.
        """
        df = pd.read_csv(document_filename)
        content = {}
        for index,row in df.iterrows():
            course_detail = self.cleanData(row['description']+' '+row['content'] +' ' +row['coursename'])
            title = row['courseno'] + ' - ' + row['coursename']
            content[title] = course_detail
        return content
    
    
    def readDictionary(self, base_directory):
        """
            Input   : directory path.
            Returns : Wikipedia article titles and contents ordered list.
        """
        content = {}

        data = json.load(open(base_directory,"r",encoding="utf-8"))
        for title in data:
            content[title] = self.cleanData(data[title] + title)
        return content