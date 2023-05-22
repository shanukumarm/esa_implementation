from ESA import ESA
from Preprocessing import Preprocessing
from Evaluation import Evaluation
from Visualized_evaluation import Visualized_evaluation

from sklearn.metrics.pairwise import cosine_similarity        # FOR COSINE SIMILARITY MEASURE


class Calling:
    def __init__(self, base_path, document_path, eval_path, wiki_or_not, phrases, abbreviations):
        self.base_path = base_path
        self.document_path = document_path
        self.eval_path = eval_path
        self.wiki_or_not = wiki_or_not
        self.phrases = phrases
        self.abbreviations = abbreviations
        self.dept = 'CS'
        
    def call(self):
        self.preprocessing = Preprocessing(self.phrases, self.abbreviations)
        self.document_content = self.preprocessing.readCourseData(self.document_path)
        if self.wiki_or_not == "wiki":
            self.base_content = self.preprocessing.readWikipediaData(self.base_path)
        else:
            self.base_content = self.preprocessing.readDictionary(self.base_path)
        
        self.esa = ESA(self.base_content, self.document_content, self.document_content, "same")
        self.document_esa_matrix, self.query_esa_matrix, self.esa_similarity_matrix = self.esa.computation() 
        
        self.titles, _ = self.esa.dict_to_list(self.document_content)
        self.evaluation = Evaluation(self.eval_path, self.titles, self.esa_similarity_matrix, self.dept)
        self.new_esa_similarity_matrix, self.evaluation_matrix = self.evaluation.evaluate()
        
        self.score = 0.7
        self.visualization = Visualized_evaluation(self.new_esa_similarity_matrix, self.evaluation_matrix, self.titles, self.dept, self.score)
        self.visualization.plot()


phrases = []
abbreviations = {}

# Using Wikipedia
wikipedia_articles_path = "./codes/dataset/QueriedWikipediaArticles/"
course_detail_path = "./codes/dataset/CourseData.csv"
eval_matrix_path = "./codes/dataset/Revised_EvaluationData.txt"

calling = Calling(wikipedia_articles_path, course_detail_path, eval_matrix_path, "wiki", phrases, abbreviations)
calling.call()


# Using IISC
iisc_path = "./codes/dataset/iisc.json"
course_detail_path = "./codes/dataset/CourseData.csv"
eval_matrix_path = "./codes/dataset/Revised_EvaluationData.txt"

calling = Calling(iisc_path, course_detail_path, eval_matrix_path, "iisc", phrases, abbreviations)
calling.call()


# Using IITD
iitd_path = "./codes/dataset/iitd.json"
course_detail_path = "./codes/dataset/CourseData.csv"
eval_matrix_path = "./codes/dataset/Revised_EvaluationData.txt"

calling = Calling(iitd_path, course_detail_path, eval_matrix_path, "iitd", phrases, abbreviations)
calling.call()


