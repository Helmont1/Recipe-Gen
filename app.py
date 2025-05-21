import streamlit as st
import asyncio
import os
from dotenv import load_dotenv
from src.api.spoonacular_integration import fetch_recipes_by_ingredients
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Qdrant
from langchain_openai import ChatOpenAI
from langchain.docstore.document import Document
import json

os.environ["LANGCHAIN_ALLOW_DANGEROUS_DESERIALIZATION"] = "true"
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

embedder = OpenAIEmbeddings(openai_api_key=openai_api_key)

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())
st.title("Gerador de Receitas com RAG")
st.markdown(
    "> ⚠️ **Aviso:** Esta IA pode cometer erros. Sempre revise as receitas geradas antes de utilizá-las."
)
ingredients = st.text_input("Insira os ingredientes (separados por vírgula):")
if ingredients:
    ingredients_list = [x.strip() for x in ingredients.split(",")]

    recipe_docs = fetch_recipes_by_ingredients(ingredients_list)
    if not recipe_docs:
        llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            model="gpt-4.1-nano",
            temperature=0.7,
            max_tokens=1024
        )
        generated_recipes = []
        for i in range(5):
            fallback_prompt = f"""
            Create a detailed recipe using only these ingredients: {', '.join(ingredients_list)}.
            Do not add any other ingredients. If a required ingredient is missing, mention it in the instructions, but do not invent or add new ones.
            Include ingredient quantities and step-by-step instructions.
            Give the recipe a creative name.
            """
            recipe = llm.invoke(fallback_prompt)
            recipe_content = recipe.content if hasattr(
                recipe, 'content') else str(recipe)
            generated_recipes.append(recipe_content)
        recipe_docs = [Document(page_content=content, metadata={
                                "id": f"openai_fallback_{i+1}"}) for i, content in enumerate(generated_recipes)]
        Qdrant.from_documents(
            recipe_docs,
            embedder,
            url=qdrant_url,
            api_key=qdrant_api_key,
            collection_name="recipes"
        )
        st.success("Receitas armazenadas no Qdrant!")

    else:
        st.info(f"Capturadas {len(recipe_docs)} receitas do Spoonacular.")

        Qdrant.from_documents(
            recipe_docs,
            embedder,
            url=qdrant_url,
            api_key=qdrant_api_key,
            collection_name="recipes"
        )
        st.success("Receitas armazenadas no Qdrant!")

        context_prompt = f"""
        Using only these ingredients: {', '.join(ingredients_list)}, and considering the following recipes as inspiration: {[doc.page_content for doc in recipe_docs]}, create a new, unique recipe. Do not add any other ingredients. Include ingredient quantities (in grams) and step-by-step instructions. If a required ingredient is missing, mention it in the instructions, but do not invent or add new ones.
        Return the response as a structured JSON with the following format:
        {{
            "title": "...",
            "ingredients": ["ingredient1", "ingredient2", ...],
            "steps": ["step1", "step2", ...],
            "nutrition_markdown": "| Nutriente | Valor |\n|---|---|\n| Calorias | ... | ... | ... | ... |"
        }}
        The recipe and all fields must be in Portuguese. The nutrition information must be a markdown table in the field 'nutrition_markdown'.
        """
        llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            model="gpt-4.1-nano",
            temperature=0.7,
            max_tokens=1024
        )
        final_recipe = llm.invoke(context_prompt)
        st.write("### Sua Receita Personalizada ")
        try:
            recipe_json = json.loads(final_recipe.content if hasattr(
                final_recipe, 'content') else final_recipe)
            st.subheader(recipe_json.get("title", "Receita"))
            st.markdown("**Ingredientes:**")
            for ingredient in recipe_json.get("ingredients", []):
                st.write(f"- {ingredient}")
            st.markdown("**Modo de preparo:**")
            for i, step in enumerate(recipe_json.get("steps", []), 1):
                st.write(f"{i}. {step}")
            st.markdown("**Informação nutricional:**")
            nutrition_md = recipe_json.get("nutrition_markdown", "")
            if nutrition_md:
                st.markdown(nutrition_md)
        except Exception as e:
            st.write(final_recipe.content if hasattr(
                final_recipe, 'content') else final_recipe)
