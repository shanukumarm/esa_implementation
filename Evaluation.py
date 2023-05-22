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




class Evaluation:
    def __init__(self, filename, titles, similarity_matrix, dept):
        self.filename = filename
        self.titles = titles
        self.num_courses = len(titles)
        self.similarity_matrix = similarity_matrix
        self.dept = dept
        
    def createEvaluationMatrix(self):
        """
            Input   : Path to evaluation data, number of courses, list of course_titles
            Returns : A binary Relevance feedback matrix.
        """
        with open(self.filename) as file:
            eval_data = file.read().strip()
            eval_data = eval(eval_data)

        self.eval_matrix = np.zeros((self.num_courses, self.num_courses), dtype=int)
        np.fill_diagonal(self.eval_matrix, 1)

        for pair in eval_data:
            index_1  = self.titles.index(pair[0])
            index_2  = self.titles.index(pair[1])
            self.eval_matrix[index_1][index_2] = 1

        return self.eval_matrix

    def getPrunedMatrix(self, matrix):
        """
            Input   : matrix, list of courses, department code.
            Returns : matrix - department x all departments.
        """
        min_index = 0
        max_index = 0
        for i in range(len(self.titles)):
            if(self.dept in self.titles[i] and min_index==0):
                min_index = i
            elif(self.dept in self.titles[i] and min_index!=0):
                max_index = i
            elif(min_index!=0 and max_index!=0):
                break
        return matrix[min_index : max_index+1]

    def getNDCG(self, a, b):
        """
            Input   : The obtained rankings, and expected relevance.
            Returns : nDCG value.
        """
        dcg = 0
        for i in range(1,len(a)):
            dcg += b[a[i]]/math.log2(i+1)

        idcg = 0
        num_relevant = int(np.sum(b))
        for i in range(1,num_relevant):
            idcg += 1/math.log2(i+1)

        return dcg/idcg

    def evaluation(self):
        """
            Input   : The similarity matrix and evaluation matrix.
            Returns : Course wise nDCG values and average nDCG values.
        """
        ndcg_vals = []
        num_courses = int(len(self.eval_matrix.T[0]))
        for i in range(num_courses):
            ranked_course_indices = np.argsort(-1*self.similarity_matrix[i])
            ndcg_vals.append(self.getNDCG(ranked_course_indices, self.eval_matrix[i]))

        return ndcg_vals

    def printCourseWiseNDCGValues(self, ndcg_vals, titles):
        """
            Input   : Course wise nDCG values, department titles.
            Prints  : Course wise nDCG values, average nDCG vale.
        """
        for i in range(len(titles)):
            print(titles[i]," - ",ndcg_vals[i])
        print("\n Average nDCG = ",sum(ndcg_vals)/len(titles))
        
        
    def evaluate(self):
        self.eval_matrix = self.createEvaluationMatrix()
        cs_titles = self.getPrunedMatrix(self.titles)
        
        self.eval_matrix = self.getPrunedMatrix(self.eval_matrix.T).T
        self.eval_matrix = self.getPrunedMatrix(self.eval_matrix)
        self.similarity_matrix = self.getPrunedMatrix(self.similarity_matrix.T).T
        self.similarity_matrix = self.getPrunedMatrix(self.similarity_matrix)
        
        ndcg_vals = self.evaluation()
        self.printCourseWiseNDCGValues(ndcg_vals, cs_titles)
        print(np.nansum(ndcg_vals)/(len(ndcg_vals)-1))
        return self.similarity_matrix, self.eval_matrix