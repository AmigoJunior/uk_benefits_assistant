import streamlit as st
import time
import json
import pandas as pd

from tqdm.auto import tqdm
from openai import OpenAI
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from index_docs import create_index, index_documents, load_documents


# Load Sentence Transformer model for embedding
@st.cache_resource
def load_model():
    return SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

# Connect to Elasticsearch
es_client = Elasticsearch('http://localhost:9200')

model = load_model()
# Load documents
# with open('../data/document-with-ids.json', 'r') as file:
#     documents = json.load(file)

# Load ground truth data
df_ground_truth = pd.read_csv('../data/ground-truth-data.csv')
ground_truth = df_ground_truth.to_dict(orient='records')

# Load Sentence Transformer model for embedding
# model_name = 'multi-qa-MiniLM-L6-cos-v1'
# model = SentenceTransformer(model_name)



def elastic_search_knn(field, vector, section, index_name="benefit-claims"):
    knn = {
        "field": field,
        "query_vector": vector,
        "k": 5,
        "num_candidates": 10000,
        "filter": {
            "term": {
                "section": section
            }
        }
    }

    search_query = {
        "knn": knn,
        "_source": ["answer", "section", "question", "category", "id"]
    }

    es_results = es_client.search(
        index=index_name,
        body=search_query
    )
    
    result_docs = []
    
    for hit in es_results['hits']['hits']:
        result_docs.append(hit['_source'])

    return result_docs

def question_answer_vector_knn(q):
    question = q['question']
    section = q['section']

    v_q = model.encode(question)

    return elastic_search_knn('question_answer_vector', v_q, section)


# Function to retrieve the vector for a question and search with Elasticsearch
def question_answer_vector_knn(q):
    question = q['question']
    section = q['section']

    # Get the embedding for the question using Sentence Transformer
    v_q = model.encode(question)

    # Perform Elasticsearch search
    return elastic_search_knn('question_answer_vector', v_q, section)

# Function to build the prompt for the LLM
def build_prompt(query, search_results):
    prompt_template = """
You are an expert in United Kingdom Benefit Claims and Medical Negligence Claims. Answer the QUESTION based on the CONTEXT from 
the FAQ databases of Benefits database and NHS claims management. 
Use only the facts from the CONTEXT when answering the QUESTION.

QUESTION: {question}

CONTEXT: 
{context}
""".strip()

    context = ""
    for doc in search_results:
        context += f"category: {doc['category']}\nquestion: {doc['question']}\nanswer: {doc['answer']}\nsection: {doc['section']}\n\n"
    
    prompt = prompt_template.format(question=query, context=context).strip()
    return prompt

# Initialize OpenAI client
client = OpenAI()

# Function to send the prompt to the LLM
def llm(prompt, model='gpt-4o-mini'):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# RAG function that retrieves documents from Elasticsearch and generates the answer using LLM
def rag_function(query: dict, model='gpt-4o-mini') -> str:
    search_results = question_answer_vector_knn(query)
    prompt = build_prompt(query['question'], search_results)
    answer = llm(prompt, model=model)
    return answer

def initialize_index():
    # Check if index exists and create if it doesn't
    index_created = create_index(es_client)
    
    if index_created:
        # Load and index documents
        documents = load_documents('../data/document-with-ids.json')
        index_documents(es_client, documents, model)
        st.success("Index created and documents indexed successfully!")
    else:
        st.info("Index already exists. Using existing index.")

# Streamlit Application
def main():
    st.title("RAG Application")

    # Input box
    user_input = st.text_input("Enter your query:")

    # Section input (since section is required for the query)
    section_input = st.text_input("Enter the section (e.g., benefits, nhs-claims):")

    # Ask button
    if st.button("Ask"):
        if user_input and section_input:
            with st.spinner("Processing..."):
                # Prepare the query as a dictionary (including section)
                query = {"question": user_input, "section": section_input}
                result = rag_function(query)
            st.success("Done!")
            st.write("Result:", result)
        else:
            st.warning("Please enter both a query and a section.")

if __name__ == "__main__":
    main()
