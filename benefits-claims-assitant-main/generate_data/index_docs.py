import json
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Load Sentence Transformer model for embedding
model_name = 'multi-qa-MiniLM-L6-cos-v1'
model = SentenceTransformer(model_name)

# Connect to Elasticsearch
es_client = Elasticsearch('http://localhost:9200')

def create_index(es_client, index_name="benefit-claims"):
    index_settings = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0
        },
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
                    "similarity": "cosine"
                },
            }
        }
    }

    if not es_client.indices.exists(index=index_name):
        es_client.indices.create(index=index_name, body=index_settings)
        print(f"Index '{index_name}' created successfully.")
    else:
        print(f"Index '{index_name}' already exists.")


def index_documents(es_client, documents, model, index_name="benefit-claims"):
    for doc in tqdm(documents, desc="Indexing documents"):
        doc_vector = model.encode(doc['question'] + " " + doc['answer'])
        doc['question_answer_vector'] = doc_vector.tolist()
        es_client.index(index=index_name, body=doc, id=doc['id'])
    print(f"Indexed {len(documents)} documents.")

def load_documents(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)