import streamlit as st
import asyncio
import os
from dotenv import load_dotenv
from src.api.spoonacular_integration import fetch_recipes_by_ingredients
from langchain.embeddings import OpenAIEmbeddings
from src.rag.vectorstore import store_recipes, retrieve_similar_recipes
from src.rag.llm import generate_recipe
from src.rag.utils import parse_ingredients


def initialize_environment():
    """Initialize environment variables and configurations"""
    os.environ["LANGCHAIN_ALLOW_DANGEROUS_DESERIALIZATION"] = "true"
    load_dotenv()

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    return {
        'openai_api_key': os.getenv("OPENAI_API_KEY"),
        'qdrant_url': os.getenv("QDRANT_URL"),
        'qdrant_api_key': os.getenv("QDRANT_API_KEY"),
        'collection_name': "recipes",
        'embedder': OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    }


def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'generated_recipes' not in st.session_state:
        st.session_state.generated_recipes = []
    if 'ingredients_cache' not in st.session_state:
        st.session_state.ingredients_cache = None


def should_reset_cache(current_query):
    """Check if cache should be reset based on ingredients change"""
    if st.session_state.ingredients_cache != current_query:
        st.session_state.generated_recipes = []
        st.session_state.ingredients_cache = current_query
        return True
    return False


def fetch_recipes_with_cache(ingredients_list, config):
    """Fetch recipes using Qdrant cache with Spoonacular fallback"""
    query = ", ".join(ingredients_list)

    # Try to retrieve from Qdrant first
    try:
        retrieved_docs = retrieve_similar_recipes(
            query, config['embedder'], config['qdrant_url'],
            config['qdrant_api_key'], config['collection_name'], k=10
        )
    except Exception as e:
        st.warning(f"Erro ao buscar no Qdrant: {str(e)}")
        retrieved_docs = []

    # If no docs found, fetch from Spoonacular and store
    if not retrieved_docs:
        recipe_docs = fetch_recipes_by_ingredients(ingredients_list)
        print(
            f"Receitas do Spoonacular: {len(recipe_docs) if recipe_docs else 0}")

        if not recipe_docs:
            st.error("Nenhuma receita encontrada com esses ingredientes.")
            return None

        # Store in Qdrant for future use
        store_recipes(recipe_docs, config['embedder'], config['qdrant_url'],
                      config['qdrant_api_key'], config['collection_name'])

        # Retrieve the newly stored recipes
        retrieved_docs = retrieve_similar_recipes(
            query, config['embedder'], config['qdrant_url'],
            config['qdrant_api_key'], config['collection_name'], k=10
        )

        if not retrieved_docs:
            st.error(
                "Receitas armazenadas, mas nenhuma encontrada na busca. Tente novamente.")
            return None
        else:
            st.info(
                f"{len(retrieved_docs)} receitas similares encontradas no Qdrant.")

    return retrieved_docs


def generate_new_recipe(ingredients_list, context, openai_api_key):
    """Generate a new recipe and add to session state"""
    recipe_json = generate_recipe(context, ingredients_list, openai_api_key)
    st.session_state.generated_recipes.append(recipe_json)
    return recipe_json


def render_recipe_interface(ingredients_list, context, openai_api_key):
    """Render the recipe generation and selection interface"""
    # Generate first recipe if none exists
    if len(st.session_state.generated_recipes) == 0:
        generate_new_recipe(ingredients_list, context, openai_api_key)

    # Button to generate another recipe
    if st.button("Gerar outra receita"):
        generate_new_recipe(ingredients_list, context, openai_api_key)

    # Recipe selection dropdown
    if st.session_state.generated_recipes:
        opcoes = [
            f"Receita {i+1}" for i in range(len(st.session_state.generated_recipes))]
        idx = st.selectbox(
            "Escolha a receita para visualizar:",
            options=range(len(opcoes)),
            format_func=lambda i: opcoes[i]
        )

        recipe_json = st.session_state.generated_recipes[idx]
        render_recipe_display(recipe_json)


def render_recipe_display(recipe_json):
    """Render the selected recipe details"""
    st.write("### Sua Receita Personalizada")

    if isinstance(recipe_json, dict):
        # Recipe title
        st.subheader(recipe_json.get("title", "Receita"))

        # Ingredients section
        st.markdown("**Ingredientes:**")
        for ingredient in recipe_json.get("ingredients", []):
            st.write(f"- {ingredient}")

        # Instructions section
        st.markdown("**Modo de preparo:**")
        for i, step in enumerate(recipe_json.get("steps", []), 1):
            st.write(f"{i}. {step}")

        # Nutrition information
        st.markdown("**Informação nutricional:**")
        nutrition_md = recipe_json.get("nutrition_markdown", "")
        if nutrition_md:
            st.markdown(nutrition_md)

        # Seasoning suggestions
        sugestoes_temperos = recipe_json.get("sugestoes_temperos", [])
        if sugestoes_temperos:
            st.markdown("**Sugestões de temperos/ervas/especiarias:**")
            for sugestao in sugestoes_temperos:
                st.write(f"- {sugestao}")
    else:
        st.write(recipe_json)


def main():
    """Main application function"""
    # Initialize environment and session state
    config = initialize_environment()
    initialize_session_state()

    # Streamlit UI
    st.title("Gerador de Receitas com RAG")
    ingredients = st.text_input(
        "Insira os ingredientes (separados por vírgula):")

    if ingredients:
        ingredients_list = parse_ingredients(ingredients)
        query = ", ".join(ingredients_list)

        # Reset cache if ingredients changed
        should_reset_cache(query)

        # Fetch recipes with caching logic
        retrieved_docs = fetch_recipes_with_cache(ingredients_list, config)

        if retrieved_docs:
            context = [doc.page_content for doc in retrieved_docs]
            render_recipe_interface(
                ingredients_list, context, config['openai_api_key'])


if __name__ == "__main__":
    main()
