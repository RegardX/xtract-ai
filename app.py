import pandas as pd
from flask import Flask, jsonify, request
from transformers import (AutoModel, AutoModelWithLMHead, AutoTokenizer, 
                          pipeline, TFMarianMTModel)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

# Logger configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
DATA = pd.read_csv("policies_procedures_data.csv")
MODEL_PATHS = {
    'language_detection': "models/xlm-roberta-base-language-detection",
    'translation': "models/opus-mt-tr-en",
    'qa_roberta': "models/question-answering-roberta-base-s",
    'qa_t5': "models/t5-base-finetuned-question-answering"
}

# Load models initially
class ModelLoader:
    def __init__(self, model_paths):
        self.language_detector = pipeline("text-classification", model=model_paths['language_detection'])
        self.translation_model = TFMarianMTModel.from_pretrained(model_paths['translation'])
        self.translation_tokenizer = AutoTokenizer.from_pretrained(model_paths['translation'])
        self.qa_roberta = pipeline("question-answering", model=model_paths['qa_roberta'])
        self.qa_t5_tokenizer = AutoTokenizer.from_pretrained(model_paths['qa_t5'])
        self.qa_t5_model = AutoModelWithLMHead.from_pretrained(model_paths['qa_t5'])

    def detect_language(self, text):
        return self.language_detector(text)[0]["label"]

    def translate(self, text):
        batch = self.translation_tokenizer([text], return_tensors="tf")
        gen = self.translation_model.generate(**batch)
        return self.translation_tokenizer.batch_decode(gen, skip_special_tokens=True)[0]

    def extractive_qa(self, question, context):
        return self.qa_roberta(question=question, context=context)

    def generative_qa(self, question, context):
        whole_text = f"question: {question} context: {context}"
        encoded_input = self.qa_t5_tokenizer([whole_text], return_tensors='pt', max_length=512, truncation=True)
        output = self.qa_t5_model.generate(input_ids=encoded_input.input_ids, attention_mask=encoded_input.attention_mask)
        return self.qa_t5_tokenizer.decode(output[0], skip_special_tokens=True)

models = ModelLoader(MODEL_PATHS)

# Relevance computation function
def compute_relevance(query):
    documents = DATA.Text.tolist()
    authors = DATA.Names.tolist()
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    query_vector = vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix)
    most_relevant_idx = cosine_similarities.argmax()
    return documents[most_relevant_idx], authors[most_relevant_idx]

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
        context, author = compute_relevance(query)
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


