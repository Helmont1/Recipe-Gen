from langchain.chains import RetrievalQA
from langchain.vectorstores import Qdrant
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
import os

# Aprimoramento de Grandes Modelos de Linguagem (LLM) por
# Recuperação e Geração de Respostas (RAG) na Criação de Receitas Culinárias Contextualizadas
from src.api.spoonacular_integration import fetch_recipes_by_ingredients

openai_api_key = os.getenv("OPENAI_API_KEY")
embedder = OpenAIEmbeddings(openai_api_key=openai_api_key)

load_dotenv()


def create_rag_pipeline(vector_db):
    """Create fresh RAG chain with Spoonacular recipes"""
    llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        model="gpt-4.1-nano",
        temperature=0.7,
        max_tokens=1024
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

    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    vector_db = Qdrant.from_documents(
        recipe_docs, embedder, url=qdrant_url, api_key=qdrant_api_key, collection_name="recipes")

    rag_chain = create_rag_pipeline(vector_db)
    rag_chain.retriever = vector_db.as_retriever(search_kwargs={"k": 2})

    prompt = f"""
    Create a detailed recipe using primarily these ingredients: {', '.join(ingredients)}.
    Consider these similar recipes for inspiration: {[doc.page_content for doc in recipe_docs]}.
    Include ingredient quantities and step-by-step instructions.
    Return the recipe with nutrition information. and generate the recipe in portuguese.
    """

    resposta_teste = rag_chain.run(
        "Crie uma receita detalhada usando tomate e queijo.")
    print("Resposta teste:", resposta_teste)

    return rag_chain.run(prompt)
