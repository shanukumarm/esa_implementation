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


from Evaluation import Evaluation


import matplotlib.pyplot as plt
from matplotlib import colors

class Visualized_evaluation:
    def __init__(self, similarity_matrix, evaluation_matrix, titles, dept, score):
        self.similarity_matrix = similarity_matrix
        self.evaluation_matrix = evaluation_matrix
        self.dept = dept
        self.titles = titles
        self.score = score
        
    def ranking_matrix_creation(self):
        ranking_matrix = np.zeros(self.similarity_matrix.shape, dtype=int)
        for index, similarity_vector in enumerate(self.similarity_matrix):
            ranking_matrix[index] = np.argsort(-1*similarity_vector)
        return ranking_matrix
    
    def ploting(self, displaying_matrix):
        cs_titles = Evaluation("", self.titles, [], self.dept).getPrunedMatrix(self.titles)

        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(111)
        cmap = colors.ListedColormap(['white', 'yellow'])
        
        plt.imshow(self.evaluation_matrix, cmap=cmap, interpolation='nearest')

        for (j, label), i in np.ndenumerate(displaying_matrix):
            if self.evaluation_matrix[i][j] == 1 and self.similarity_matrix[i][j] >= self.score:
                ax.text(i, j, label+1, color='green', ha='center', va='center')
            elif self.evaluation_matrix[i][j] == 1 and self.similarity_matrix[i][j] < self.score:
                ax.text(i, j, label+1, color='red', ha='center', va='center')
            elif self.evaluation_matrix[i][j] == 0 and self.similarity_matrix[i][j] >= self.score:
                ax.text(i, j, label+1, color='red', ha='center', va='center')
            else:
                ax.text(i, j, label+1, color='green', ha='center', va='center')

        plt.xticks(range(len(cs_titles)), cs_titles, rotation='vertical')
        plt.yticks(range(len(cs_titles)), cs_titles)
        plt.show()
        
    def plot(self):
        np.fill_diagonal(self.evaluation_matrix, 0)
        np.fill_diagonal(self.similarity_matrix, 0)
        
        ranked_matrix = self.ranking_matrix_creation()
        self.ploting(ranked_matrix)