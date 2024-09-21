import os
import pandas as pd
import json
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
from tqdm.auto import tqdm
from dotenv import load_dotenv

from db import init_db

load_dotenv()

ELASTIC_URL = os.getenv("ELASTIC_URL_LOCAL")
MODEL_NAME = os.getenv("MODEL_NAME")
INDEX_NAME = os.getenv("INDEX_NAME")



def fetch_documents():
    print("Fetching documents...")
    file_path = "/workspaces/uk_benefits_assistant/benefits-claims-assitant-main/stream_app/app/document-with-ids.json"
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    with open(file_path, 'r') as file:
        documents = json.load(file)
    
    print(f"Fetched {len(documents)} documents")
    return documents


def fetch_ground_truth():
    print("Fetching ground truth data...")
    file_path = "//workspaces/uk_benefits_assistant/benefits-claims-assitant-main/stream_app/app/ground-truth-data.csv"
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    df_ground_truth = pd.read_csv(file_path)
    df_ground_truth = df_ground_truth[
        df_ground_truth.section == "general claim benefits"
    ]
    ground_truth = df_ground_truth.to_dict(orient="records")
    print(f"Fetched {len(ground_truth)} ground truth records")
    return ground_truth


def load_model():
    print(f"Loading model: {MODEL_NAME}")
    return SentenceTransformer(MODEL_NAME)


def setup_elasticsearch():
    print("Setting up Elasticsearch...")
    es_client = Elasticsearch(ELASTIC_URL)

    index_settings = {
        "settings": {"number_of_shards": 1, "number_of_replicas": 0},
        "mappings": {
            "properties": {
                "answer": {"type": "text"},
                "category": {"type": "text"},
                "question": {"type": "text"},
                "section": {"type": "keyword"},
                "id": {"type": "keyword"},
                "question_answer_vector": {
                    "type": "dense_vector",
                    "dims": 384,
                    "index": True,
                    "similarity": "cosine",
                },
            }
        },
    }

    es_client.indices.delete(index=INDEX_NAME, ignore_unavailable=True)
    es_client.indices.create(index=INDEX_NAME, body=index_settings)
    print(f"Elasticsearch index '{INDEX_NAME}' created")
    return es_client


def index_documents(es_client, documents, model):
    print("Indexing documents...")
    for doc in tqdm(documents):
        question = doc["question"]
        answer = doc["answer"]
        doc["question_answer_vector"] = model.encode(question + " " + answer).tolist()
        es_client.index(index=INDEX_NAME, document=doc)
    print(f"Indexed {len(documents)} documents")


def main():
    print("Starting the indexing process...")

    documents = fetch_documents()
    ground_truth = fetch_ground_truth()
    model = load_model()
    es_client = setup_elasticsearch()
    index_documents(es_client, documents, model)

    print("Initializing database...")
    init_db()

    print("Indexing process completed successfully!")


if __name__ == "__main__":
    main()