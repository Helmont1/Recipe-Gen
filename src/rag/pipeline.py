from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_db = FAISS.load_local("data/faiss_index", embedder)

llm = HuggingFaceHub(
    repo_id="meta-llama/Llama-2-7b-chat-hf",
    model_kwargs={"temperature": 0.5, "max_length": 512}
)

rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vector_db.as_retriever(search_kwargs={"k": 3}),
    chain_type="stuff"
)

def generate_recipe(ingredients: list):
    recipes = fetch_recipes_by_ingredients(ingredients)
    context = rag_chain.run(f"Suggest a recipe using: {', '.join(ingredients)}")
    return context