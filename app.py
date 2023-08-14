import pandas as pd
import numpy as np
from typing import List
from flask import Flask, jsonify, request
from transformers import (AutoModel, AutoModelWithLMHead, AutoTokenizer, 
                          pipeline)

from sklearn.metrics.pairwise import cosine_similarity
import logging
from sentence_transformers import SentenceTransformer
import faiss

# Logger configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
DATA = pd.read_csv("policies_procedures_data.csv")
MODEL_PATHS = {
    'language_detection': "models/xlm-roberta-base-language-detection",
    'translation': "Helsinki-NLP/opus-mt-tc-big-tr-en",
    'qa_t5': "models/t5-base-finetuned-question-answering"
}

# Load models initially
class ModelLoader:
    def __init__(self, model_paths):
        self.language_detector = pipeline("text-classification", model=model_paths['language_detection'])
        self.translation_model = pipeline("translation", model=model_paths['translation'])
        self.translation_tokenizer = AutoTokenizer.from_pretrained(model_paths['translation'])
        self.qa_t5_tokenizer = AutoTokenizer.from_pretrained(model_paths['qa_t5'])
        self.qa_t5_model = AutoModelWithLMHead.from_pretrained(model_paths['qa_t5'])

    def detect_language(self, text):
        return self.language_detector(text)[0]["label"]

    def translate(self, text):
        return self.translation_model(text)[0]['translation_text']


    def generative_qa(self, question, context):
        whole_text = f"Question: {question}, Context: {context}. Only use the context not your imagination."
        encoded_input = self.qa_t5_tokenizer([whole_text], return_tensors='pt', max_length=512, truncation=True)
        output = self.qa_t5_model.generate(input_ids=encoded_input.input_ids, attention_mask=encoded_input.attention_mask)
        return self.qa_t5_tokenizer.decode(output[0], skip_special_tokens=True)
    
    

models = ModelLoader(MODEL_PATHS)



#Relevance computation using Faiss


def compute_relevance(query):
  
    model = SentenceTransformer('sentence-transformers/paraphrase-distilroberta-base-v1')


    df = pd.DataFrame({
        'texts': DATA.Text.tolist(),
        'author' : DATA.Names.tolist()
    })

 
    embeddings = model.encode(df['texts'].tolist())

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)


    query_embedding = model.encode(query)
    query_embedding = np.array(query_embedding).reshape(1, -1)  


    D, I = index.search(query_embedding, k=1)  
    most_similar_text_index = I[0][0]
    most_similar_text = df.iloc[most_similar_text_index]['texts']
    similarity_score = 1 - D[0][0]
    return most_similar_text, df.author[most_similar_text_index], similarity_score


# Flask application
app = Flask(__name__)

@app.errorhandler(Exception)
def handle_exception(e):
    error_messages = {
        400: "BAD REQUEST",
        404: "NOT FOUND",
        405: "REQUESTED METHOD IS NOT SUPPORTED",
        503: "Service Unavailable",
        500: "INTERNAL SERVER ERROR"
    }
    status_code = e.code if isinstance(e, HTTPException) else 500
    logger.error(f"Error encountered: {error_messages.get(status_code)}")
    return jsonify({"IsSucceed": False, "ErrorCode": str(status_code), "ErrorMessage": error_messages.get(status_code, "INTERNAL SERVER ERROR")}), status_code

@app.route('/xtract', methods=['POST'])
def xtract():
    try:
        query = request.json["query"]
        language = models.detect_language(query)
        if language == "tr":
            query = models.translate(query)
        context, author, similarity = compute_relevance(query)
        
        if similarity < -60:
            return_answer = "This topic seems beyond the scope of my knowledge."
        else:
            gqa_answer = models.generative_qa(query, context)
            return_answer = gqa_answer or f"You can consult with {author} about this matter."

        return jsonify({
            "Result": return_answer,
            "Author": author,
            "IsSucceed": True,
            "ErrorCode": "",
            "ErrorMessage": ""
        })
    except Exception as e:
        logger.error(f"Error while processing xtract endpoint: {str(e)}")
        return handle_exception(e)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=6000, debug=True)


