from langchain_openai import OpenAIEmbeddings
from langchain.document_loaders import JSONLoader
from langchain.vectorstores import Qdrant
import os

loader = JSONLoader(file_path='data/recipes.json', jq_schema='.[]')
recipes = loader.load()

openai_api_key = os.getenv("OPENAI_API_KEY")
embedder = OpenAIEmbeddings(openai_api_key=openai_api_key)
recipe_texts = [doc.page_content for doc in recipes]
embeddings = embedder.embed_documents(recipe_texts)

qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")
vector_db = Qdrant.from_documents(
    recipes, embedder, url=qdrant_url, api_key=qdrant_api_key, collection_name="recipes")
