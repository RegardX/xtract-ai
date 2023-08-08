import pandas as pd
from flask import Flask, render_template, request, jsonify
import werkzeug
from flask import send_file
from werkzeug.exceptions import HTTPException
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers  import  AutoTokenizer, AutoModel, pipeline, AutoModelWithLMHead,TFMarianMTModel
from typing import List
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

import glob
import gc
import requests

data = pd.read_csv("policies_procedures_data.csv")

def relevance(query):


    # Example document list
    documents =  data.Text.to_list()
    authors = data.Names.to_list()

    # Create a TfidfVectorizer and fit it on your documents
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)

    # Transform your query using the fitted vectorizer
    query_vector = vectorizer.transform([query])

    # Calculate the cosine similarity between the query and each document
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix)

    # Find the index of the most relevant document
    most_relevant_idx = cosine_similarities.argmax()

    # Get the most relevant document based on the index
    most_relevant_document = documents[most_relevant_idx]
    authors_of_document = authors[most_relevant_idx]
    # Print the most relevant document and its author

    return most_relevant_document,authors_of_document



def lang_detect(text):
    # Download pytorch model
    model_checkpoint = "papluca/xlm-roberta-base-language-detection"
    # download pre-trained language detection model
    model = pipeline(
        "text-classification",
        model=model_checkpoint,
    )


    return model(text)[0]["label"]


def translate(sample_text):
    
    model_name = f"Helsinki-NLP/opus-mt-tc-big-tr-en"

    model = TFMarianMTModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    batch = tokenizer([sample_text], return_tensors="tf")
    gen = model.generate(**batch)
    result = tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
    return result


def xtractive_qa(question,context):
    model_checkpoint = "consciousAI/question-answering-roberta-base-s"

    question_answerer = pipeline("question-answering", model=model_checkpoint)
    return question_answerer(question=question, context=context)["score"],question_answerer(question=question, context=context)["answer"]


def t5_qa(question,context):
    model_name = "MaRiOrOsSi/t5-base-finetuned-question-answering"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelWithLMHead.from_pretrained(model_name)


    whole_text = f"question: {question} context: {context}"
    encoded_input = tokenizer([whole_text],
                                 return_tensors='pt',
                                 max_length=512,
                                 truncation=True)
    output = model.generate(input_ids = encoded_input.input_ids,
                                attention_mask = encoded_input.attention_mask)
    output = tokenizer.decode(output[0], skip_special_tokens=True)
    return output





server = Flask(__name__) 

server.config['TRAP_HTTP_EXCEPTIONS']=True

@server.errorhandler(Exception)
def handle_error(e):
    try:
        if e.code < 400:
            return flask.Response.force_type(e, flask.request.environ)
        elif e.code == 400:
            return jsonify({
                        "IsSucceed": False,
                        "ErrorCode": "400",
                        "ErrorMessage": "BAD REQUEST"
                        }), 400
        
        elif e.code == 404:
            return jsonify({
                        "IsSucceed": False,
                        "ErrorCode": "404",
                        "ErrorMessage": "NOT FOUND"
                        }), 404
        
        elif e.code == 405:
            return jsonify({
                        "IsSucceed": False,
                        "ErrorCode": "405",
                        "ErrorMessage": "REQUESTED METHOD IS NOT SUPPORTED"
                        }), 405
        
        elif e.code == 503:
            return jsonify({
                        "IsSucceed": False,
                        "ErrorCode": "503",
                        "ErrorMessage": "Service Unavailable"
                        }), 503
        raise e
    except Exception as e:
        return jsonify({
                        "IsSucceed": False,
                        "ErrorCode": "500",
                        "ErrorMessage": "INTERNAL SERVER ERROR"
                        }), 500


@server.route('/xtract',methods = ['POST']) 
def xtract():
    request_ = request.json
    query = request_["query"]
    print("data loaded")
    try:

        if lang_detect(query) == "tr":
            eng_query = translate(query)
            context,author = relevance(eng_query)[0],relevance(eng_query)[1]
            score, answer = xtractive_qa(eng_query,context)
            if score < 0.50:
                gqa = t5_qa(eng_query,context)
                if gqa == '':

                    return jsonify({"Result": "You can consult with {} about this matter.".format(author),
                                    "IsSucceed": True,
                                    "ErrorCode": "",
                                    "ErrorMessage": ""
                                    })
                else:
                    return jsonify({"Result": gqa,
                                "IsSucceed": True,
                                "ErrorCode": "",
                                "ErrorMessage": ""
                                })
                   
            else:
                return jsonify({"Result": answer,
                                "IsSucceed": True,
                                "ErrorCode": "",
                                "ErrorMessage": ""
                                })
        else:
           
            context,author = relevance(query)[0],relevance(query)[1]
            score, answer = xtractive_qa(query,context)
            if score < 0.50:
                gqa = t5_qa(query,context)
                if gqa == '':
                    return jsonify({"Result": "You can consult with {} about this matter.".format(author),
                                    "IsSucceed": True,
                                    "ErrorCode": "",
                                    "ErrorMessage": ""
                                    })
                else:
                    return jsonify({"Result": gqa,
                                "IsSucceed": True,
                                "ErrorCode": "",
                                "ErrorMessage": ""
                                })
                   
            else:
                return jsonify({"Result": answer,
                                "IsSucceed": True,
                                "ErrorCode": "",
                                "ErrorMessage": ""
                                })
    


            


    except Exception as e:
        print("Oops!", e, "occurred.")
        #err = "OCR func error " + str(e.__class__) + " occurred."
        error = str(type(e).__name__) +  "--" + str(__file__) + "--" + str(e.__traceback__.tb_lineno)
        f = open('predict.txt', 'w')
        f.write('An exceptional thing happed - {}'.format(error))
        f.close()
     
    
    
if __name__ == '__main__':
    #from additionalfuncts import *
    server.run(host='127.0.0.1', port=6000,debug=True)
