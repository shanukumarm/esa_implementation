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
import nltk
nltk.download('stopwords')
nltk.download('punkt')

import sys
sys.path.append(".")

from Preprocessing import Preprocessing
from Evaluation import Evaluation
from Visualized_evaluation import Visualized_evaluation


class ESA:
    def __init__(self, wiki, documents, queries, mode):
        self.wiki = wiki
        self.documents = documents
        self.queries = queries
        self.mode = mode
        
    def dict_to_list(self, dicti):
        titles, contents = [], []
        for key in dicti:
            titles.append(key)
            contents.append(dicti[key])
        return titles, contents
        
    def document_indexing(self, titles):
        content2index = {}
        index2content = {}
        for i in range(len(titles)):
            content2index[titles[i]] = i
            index2content[i] = titles[i]
        return content2index, index2content
    
    def getTFIDFModel(self, documents):
        """
            Input   : list of document contents.
            Returns : TF-IDF Model and TF-IDF Matrix.
        """
        tfidf_vectorizer = TfidfVectorizer(stop_words='english',use_idf=True)
        tfidf_vectorizer.fit(documents)
        document_tfidf_matrix = tfidf_vectorizer.transform(documents)
        vocab = tfidf_vectorizer.vocabulary_
        return document_tfidf_matrix, vocab
    
    def esa_matrix_computation(self, titles, details, vocab, document_matrix):
        num_courses = len(titles)
        esa_matrix = np.zeros((num_courses, self.num_concepts), dtype=float)
        words_not_in_wiki = []
        for i in range(len(titles)):
            i_content = set(details[i].split())
            for word in i_content:
                if word in self.wiki_vocab:
                    word_index_in_documents = vocab[word]
                    word_index_in_wiki    = self.wiki_vocab[word]
                    documents_tf_idf        = document_matrix[word_index_in_documents, i]
                    esa_matrix[i] += documents_tf_idf * self.term_concept_matrix.T[word_index_in_wiki]
                else:
                    words_not_in_wiki.append(word)
        return esa_matrix
    
    def computation(self):
        wiki_titles, wiki_contents = self.dict_to_list(self.wiki)
#         concept2index, index2concept = self.document_indexing(wiki_titles)
        self.term_concept_matrix, self.wiki_vocab = self.getTFIDFModel(wiki_contents)
        self.num_concepts = len(wiki_titles)
        
        document_titles, document_contents = self.dict_to_list(self.documents)
#         document2index, index2document = self.document_indexing(document_titles)
        term_document_matrix, doc_vocab = self.getTFIDFModel(document_contents)
        
        document_esa_matrix = self.esa_matrix_computation(document_titles, document_contents, doc_vocab, term_document_matrix.T)
        
        if self.mode != "same":
            query_titles, query_contents = self.dict_to_list(self.queries)
            query2index, index2query = self.document_indexing(query_titles)
            term_query_matrix, query_vocab = self.getTFIDFModel(query_contents)
            query_esa_matrix = self.esa_matrix_computation(query_titles, query_contents, query_vocab, term_query_matrix.T)
            return document_esa_matrix, query_esa_matrix, cosine_similarity(document_esa_matrix, query_esa_matrix)
        
        return document_esa_matrix, document_esa_matrix, cosine_similarity(document_esa_matrix)
    
    
dept = 'CS'
phrases = []
abbreviations = {}

wikipedia_articles_path = "./codes/dataset/QueriedWikipediaArticles/"
iisc_path = "./codes/dataset/iisc.json"
iitd_path = "./codes/dataset/iitd.json"
course_detail_path = "./codes/dataset/CourseData.csv"
eval_matrix_path = "./codes/dataset/Revised_EvaluationData.txt"


preprocessing = Preprocessing(phrases, abbreviations)
wiki_content = {}
wiki_content.update(preprocessing.readWikipediaData(wikipedia_articles_path))
#wiki_content.update(preprocessing.readDictionary(iisc_path))
#wiki_content.update(preprocessing.readDictionary(iitd_path))
document_content = preprocessing.readCourseData(course_detail_path)


esa = ESA(wiki_content, document_content, document_content, "same")
document_esa_matrix, query_esa_matrix, esa_similarity_matrix = esa.computation()


titles, _ = esa.dict_to_list(document_content)

evaluation = Evaluation(eval_matrix_path, titles, esa_similarity_matrix, dept)
new_esa_similarity_matrix, evaluation_matrix = evaluation.evaluate()


score = 0.7

visualization = Visualized_evaluation(new_esa_similarity_matrix, evaluation_matrix, titles, dept, score)
visualization.plot()