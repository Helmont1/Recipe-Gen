from langchain.vectorstores import Qdrant
from langchain_qdrant import QdrantVectorStore


def get_vectorstore(embedder, url, api_key, collection_name):
    return QdrantVectorStore.from_existing_collection(
        embedding=embedder,
        collection_name=collection_name,
        url=url,
        api_key=api_key
    )


def store_recipes(recipes, embedder, url, api_key, collection_name):
    Qdrant.from_documents(
        recipes,
        embedder,
        url=url,
        api_key=api_key,
        collection_name=collection_name
    )


def retrieve_similar_recipes(query, embedder, url, api_key, collection_name, k=5):
    vector_db = get_vectorstore(embedder, url, api_key, collection_name)
    return vector_db.similarity_search(query, k=k)
