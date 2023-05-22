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


from Preprocessing import Preprocessing
from Evaluation import Evaluation
from Visualized_evaluation import Visualized_evaluation

class Jaccard:
    def __init__(self, documents, queries, mode):
        self.documents = documents
        self.queries = queries
        self.mode = mode
        
    def dict_to_list(self, dicti):
        titles, contents = [], []
        for key in dicti:
            titles.append(key)
            contents.append(dicti[key])
        return titles, contents
    
    def Intersection(self, list1, list2): 
        return list(set(list1).intersection(list2)) 

    def Union(self, list1, list2): 
        return list(set(list1).union(list2)) 

#     def jaccard(self, intersection, union):
#         intersect_score = 0
#         for term in intersection:
#             count1 = 0
#             for c in cs_courses:
#                 if term in c:
#                     count1 += 1
#             intersect_score += count1/len(titles)

#         union_score = 0
#         for term in union:
#             count2 = 0
#             for c in cs_courses:
#                 if term in c:
#                     count2 += 1
#             union_score += count2/len(titles)
#         return intersect_score/union_score

    def jaccard_similarity_matrix_computation(self):
        jac_sim = np.zeros((len(self.document_contents), len(self.query_contents)), dtype=float)
        for id1, document_content in enumerate(self.document_contents):
            document_content = list(set(document_content.split()))
            for id2, query_content in enumerate(self.query_contents):
                query_content = list(set(query_content.split()))
                intersect = self.Intersection(document_content, query_content)
                union = self.Union(document_content, query_content)
#                 jac_sim[id1][id2] = self.jaccard(intersect, union)
                jac_sim[id1][id2] = len(intersect)/len(union)
                
        return jac_sim
    
    def computation(self):
        document_titles, self.document_contents = self.dict_to_list(self.documents)
        if self.mode == "same":
            query_titles, self.query_contents = document_titles, self.document_contents
        else:
            query_titles, self.query_contents = self.dict_to_list(self.queries)
        
        similarity_matrix = self.jaccard_similarity_matrix_computation()
        return similarity_matrix
    
  
    

jaccard_matrix_path = "./codes/saved_models/jaccard_matrix.txt"

dept = 'CS'   
phrases = []
abbreviations = {}

course_detail_path = "./codes/dataset/CourseData.csv"
eval_matrix_path = "./codes/dataset/Revised_EvaluationData.txt"

preprocessing = Preprocessing(phrases, abbreviations)
document_content = preprocessing.readCourseData(course_detail_path)


jaccard = Jaccard(document_content, document_content, "same")
jaccard_similarity_matrix = jaccard.computation()
    

titles, _ = jaccard.dict_to_list(document_content)

evaluation = Evaluation(eval_matrix_path, titles, jaccard_similarity_matrix, dept)
new_jaccard_similarity_matrix, evaluation_matrix  = evaluation.evaluate()


score = 0.1

visualization = Visualized_evaluation(new_jaccard_similarity_matrix, evaluation_matrix, titles, dept, score)
visualization.plot()