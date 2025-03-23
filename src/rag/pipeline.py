import os
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv


from typing import List
import json
from src.api.spoonacular_integration import fetch_recipes_by_ingredients

embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

load_dotenv()

def create_rag_pipeline(vector_db):
    """Create fresh RAG chain with Spoonacular recipes"""
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-small",
        model_kwargs={"temperature": 0.7, "max_length": 1024}
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever()  
    )

def generate_recipe(ingredients):
    """End-to-end recipe generation with Spoonacular integration"""
    recipe_docs = fetch_recipes_by_ingredients(ingredients)
    print(recipe_docs)
    
    vector_db = FAISS.from_documents(recipe_docs, embedder)
    
    rag_chain = create_rag_pipeline(vector_db)
    rag_chain.retriever = vector_db.as_retriever(search_kwargs={"k": 2})
    
    prompt = f"""
    Create a detailed recipe using primarily these ingredients: {', '.join(ingredients)}.
    Consider these similar recipes for inspiration: {[doc.page_content for doc in recipe_docs]}.
    Include ingredient quantities and step-by-step instructions.
    """
    
    resposta_teste = rag_chain.run("Crie uma receita detalhada usando tomate e queijo.")
    print("Resposta teste:", resposta_teste)
    
    return rag_chain.run(prompt)