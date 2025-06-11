import streamlit as st
import asyncio
import os
from dotenv import load_dotenv
from src.api.spoonacular_integration import fetch_recipes_by_ingredients
from langchain.embeddings import OpenAIEmbeddings
from src.rag.vectorstore import store_recipes, retrieve_recipes_by_ingredients, retrieve_similar_recipes
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
    if 'app_logs' not in st.session_state:
        st.session_state.app_logs = []


class AppLogger:
    """Logger class to collect app messages for modal display"""

    @staticmethod
    def clear_logs():
        """Clear all logs"""
        st.session_state.app_logs = []

    @staticmethod
    def log_info(message):
        """Add info message to logs"""
        st.session_state.app_logs.append({"type": "info", "message": message})

    @staticmethod
    def log_success(message):
        """Add success message to logs"""
        st.session_state.app_logs.append(
            {"type": "success", "message": message})

    @staticmethod
    def log_warning(message):
        """Add warning message to logs"""
        st.session_state.app_logs.append(
            {"type": "warning", "message": message})

    @staticmethod
    def log_error(message):
        """Add error message to logs"""
        st.session_state.app_logs.append({"type": "error", "message": message})

    @staticmethod
    def display_logs_modal():
        """Display logs in an expandable modal"""
        if st.session_state.app_logs:
            # Create expandable section for logs
            with st.expander(f"📋 Logs ({len(st.session_state.app_logs)} mensagens)", expanded=False):
                for i, log in enumerate(st.session_state.app_logs):
                    if log["type"] == "info":
                        st.info(f"{i+1}. {log['message']}")
                    elif log["type"] == "success":
                        st.success(f"{i+1}. {log['message']}")
                    elif log["type"] == "warning":
                        st.warning(f"{i+1}. {log['message']}")
                    elif log["type"] == "error":
                        st.error(f"{i+1}. {log['message']}")
        else:
            with st.expander("📋 Logs (0 mensagens)", expanded=False):
                st.info("Nenhum log disponível ainda.")


def filter_user_ingredients_by_recipes(user_ingredients, retrieved_docs):
    """
    Filter user ingredients to only include those that appear in any of the retrieved recipes.
    Uses substring matching to handle complex ingredient names.

    Args:
        user_ingredients: List of ingredients provided by the user
        retrieved_docs: List of documents retrieved from Qdrant with metadata

    Returns:
        List of filtered ingredients that appear in the recipes
    """
    if not retrieved_docs:
        return user_ingredients

    # Get all used_ingredients from all retrieved recipes
    all_recipe_ingredients = []
    for doc in retrieved_docs:
        used_ingredients = doc.metadata.get('used_ingredients', [])
        all_recipe_ingredients.extend([ing.lower()
                                      for ing in used_ingredients])

    # Show first 10
    print(f"DEBUG: All recipe ingredients: {all_recipe_ingredients[:10]}...")

    # Filter user ingredients that are contained in any recipe ingredient
    valid_ingredients = []
    for user_ing in user_ingredients:
        user_ing_lower = user_ing.lower()

        # Check if user ingredient is contained in any recipe ingredient
        for recipe_ing in all_recipe_ingredients:
            if user_ing_lower in recipe_ing:
                print(f"DEBUG: Found match: '{user_ing}' in '{recipe_ing}'")
                valid_ingredients.append(user_ing)
                break  # Found match, move to next user ingredient

    print(f"DEBUG: Valid ingredients after filtering: {valid_ingredients}")
    return valid_ingredients


def should_reset_cache(current_query):
    """Check if cache should be reset based on ingredients change"""
    if st.session_state.ingredients_cache != current_query:
        st.session_state.generated_recipes = []
        st.session_state.ingredients_cache = current_query
        AppLogger.clear_logs()  # Clear logs for new search
        return True
    return False


def fetch_recipes_with_ingredient_filter(ingredients_list, config):
    """Fetch recipes using Spoonacular first, then fall back to Qdrant cache"""

    query = ", ".join(ingredients_list)

    # First, try to get fresh results from Spoonacular for the complete ingredient list
    AppLogger.log_info("Buscando receitas no Spoonacular...")

    try:
        print(
            f"DEBUG: Searching Spoonacular for ingredients: {ingredients_list}")
        recipe_docs = fetch_recipes_by_ingredients(ingredients_list)

        if recipe_docs:
            AppLogger.log_success(
                f"{len(recipe_docs)} receitas encontradas no Spoonacular!")

            # Show debug info about Spoonacular results
            for i, doc in enumerate(recipe_docs[:2]):
                print(f"DEBUG: Spoonacular doc {i} metadata: {doc.metadata}")

            # Store in Qdrant for future use (this will also create the index)
            AppLogger.log_info(
                "Armazenando receitas no Qdrant para futuras buscas...")
            store_recipes(recipe_docs, config['embedder'], config['qdrant_url'],
                          config['qdrant_api_key'], config['collection_name'])

            AppLogger.log_success("Receitas armazenadas com sucesso!")
            return recipe_docs

        else:
            AppLogger.log_warning(
                "Nenhuma receita encontrada no Spoonacular com todos os ingredientes.")
            AppLogger.log_info(
                "Buscando receitas similares no Qdrant...")

    except Exception as e:
        AppLogger.log_warning(f"Erro ao buscar no Spoonacular: {str(e)}")
        print(f"DEBUG: Exception in Spoonacular search: {e}")
        AppLogger.log_info("Tentando buscar no Qdrant...")

    # If Spoonacular fails or returns no results, try to retrieve from Qdrant cache
    try:
        print(f"DEBUG: Searching Qdrant for ingredients: {ingredients_list}")

        # First try with ingredient filtering
        retrieved_docs = retrieve_recipes_by_ingredients(
            ingredients_list, config['embedder'], config['qdrant_url'],
            config['qdrant_api_key'], config['collection_name'], k=5
        )

        print(
            f"DEBUG: Found {len(retrieved_docs) if retrieved_docs else 0} docs with ingredient filter")

        # Check if the cached recipes actually contain all requested ingredients
        if retrieved_docs:
            # Verify that we have recipes that use all the requested ingredients
            valid_recipes = []
            for doc in retrieved_docs:
                used_ingredients = [
                    ing.lower() for ing in doc.metadata.get('used_ingredients', [])]
                # Check if all user ingredients are found in this recipe's used ingredients
                if all(any(user_ing.lower() in recipe_ing for recipe_ing in used_ingredients)
                       for user_ing in ingredients_list):
                    valid_recipes.append(doc)

            if valid_recipes:
                AppLogger.log_success(
                    f"{len(valid_recipes)} receitas encontradas no cache local que usam todos os ingredientes!")
                return valid_recipes
            else:
                AppLogger.log_info(
                    "Receitas no cache não contêm todos os ingredientes solicitados.")

        # If no results with filter, try regular similarity search as last resort
        if not retrieved_docs:
            print("DEBUG: No results with filter, trying similarity search")
            retrieved_docs = retrieve_similar_recipes(
                query, config['embedder'], config['qdrant_url'],
                config['qdrant_api_key'], config['collection_name'], k=5
            )
            print(
                f"DEBUG: Similarity search returned {len(retrieved_docs) if retrieved_docs else 0} docs")

        if retrieved_docs:
            AppLogger.log_info(
                f"{len(retrieved_docs)} receitas similares encontradas no Qdrant.")
            AppLogger.log_warning(
                "Nota: Algumas receitas podem não usar todos os ingredientes solicitados.")
            return retrieved_docs
        else:
            AppLogger.log_error("Nenhuma receita encontrada no Qdrant.")
            return None

    except Exception as e:
        AppLogger.log_error(f"Erro ao buscar no Qdrant: {str(e)}")
        return None


def generate_new_recipe(ingredients_list, context, openai_api_key):
    """Generate a new recipe and add to session state"""
    recipe_json = generate_recipe(context, ingredients_list, openai_api_key)
    st.session_state.generated_recipes.append(recipe_json)
    return recipe_json


def render_recipe_interface(ingredients_list, context, openai_api_key):
    """Render the recipe generation and selection interface"""
    st.subheader("Gerador de Receitas")

    if len(st.session_state.generated_recipes) == 0:
        with st.spinner("Gerando sua primeira receita..."):
            generate_new_recipe(ingredients_list, context, openai_api_key)

    if st.button("Gerar outra receita"):
        with st.spinner("Criando uma nova receita..."):
            generate_new_recipe(ingredients_list, context, openai_api_key)

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
        st.subheader(recipe_json.get("title", "Receita"))

        st.markdown("**Ingredientes:**")
        for ingredient in recipe_json.get("ingredients", []):
            st.write(f"• {ingredient}")

        st.markdown("**Modo de preparo:**")
        for i, step in enumerate(recipe_json.get("steps", []), 1):
            st.write(f"{i}. {step}")

        st.markdown("**Informação nutricional:**")
        nutrition_md = recipe_json.get("nutrition_markdown", "")
        if nutrition_md:
            st.markdown(nutrition_md)

        sugestoes_temperos = recipe_json.get("sugestoes_temperos", [])
        if sugestoes_temperos:
            st.markdown("**Sugestões de temperos/ervas/especiarias:**")
            for sugestao in sugestoes_temperos:
                st.write(f"• {sugestao}")
    else:
        st.write(recipe_json)


def main():
    """Main application function"""
    config = initialize_environment()
    initialize_session_state()

    st.title("Gerador de Receitas")

    ingredients = st.text_input(
        "Insira os ingredientes em inglês (separados por vírgula):",
        placeholder="Ex: tomato,cheese,egg"
    )

    if ingredients:
        ingredients_list = parse_ingredients(ingredients)
        query = ",".join(ingredients_list)

        should_reset_cache(query)

        # Show user ingredients
        st.info(
            f"Ingredientes selecionados: **{', '.join(ingredients_list)}**")

        retrieved_docs = fetch_recipes_with_ingredient_filter(
            ingredients_list, config)

        if retrieved_docs:
            context = [doc.page_content for doc in retrieved_docs]

            # Use all the ingredients since our fetch function now ensures
            # we only get recipes that can use these ingredients
            render_recipe_interface(
                ingredients_list, context, config['openai_api_key'])
        else:
            AppLogger.log_error(
                "Não foi possível encontrar receitas com os ingredientes fornecidos.")
            AppLogger.log_info(
                "Sugestão: Tente ingredientes comuns em inglês como 'chicken', 'tomato', 'cheese', 'egg', etc.")
        AppLogger.display_logs_modal()


if __name__ == "__main__":
    main()
