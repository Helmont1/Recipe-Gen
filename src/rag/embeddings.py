from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import JSONLoader
from langchain.vectorstores import FAISS

loader = JSONLoader(file_path='data/recipes.json', jq_schema='.[]')
recipes = loader.load()

embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
recipe_texts = [doc.page_content for doc in recipes]
embeddings = embedder.embed_documents(recipe_texts)

vector_db = FAISS.from_documents(recipes, embedder)
vector_db.save_local("data/faiss_index")