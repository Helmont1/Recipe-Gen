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
        return True
    return False


def fetch_recipes_with_ingredient_filter(ingredients_list, config):
    """Fetch recipes using Qdrant with ingredient filtering and Spoonacular fallback"""

    query = ", ".join(ingredients_list)

    # Try to retrieve from Qdrant with ingredient filter
    try:
        st.info("Buscando receitas no cache local...")
        print(f"DEBUG: Searching for ingredients: {ingredients_list}")

        # First try with ingredient filtering
        retrieved_docs = retrieve_recipes_by_ingredients(
            ingredients_list, config['embedder'], config['qdrant_url'],
            config['qdrant_api_key'], config['collection_name'], k=5
        )

        print(
            f"DEBUG: Found {len(retrieved_docs) if retrieved_docs else 0} docs with ingredient filter")

        # If no results with filter, try regular similarity search
        if not retrieved_docs:
            print("DEBUG: No results with filter, trying similarity search")
            retrieved_docs = retrieve_similar_recipes(
                query, config['embedder'], config['qdrant_url'],
                config['qdrant_api_key'], config['collection_name'], k=5
            )
            print(
                f"DEBUG: Similarity search returned {len(retrieved_docs) if retrieved_docs else 0} docs")

        if retrieved_docs:
            st.success(
                f"{len(retrieved_docs)} receitas encontradas no cache local!")
            # Show some debug info about what was found
            for i, doc in enumerate(retrieved_docs[:2]):
                print(f"DEBUG: Doc {i} metadata: {doc.metadata}")
            return retrieved_docs
        else:
            st.info("Nenhuma receita encontrada no cache local.")

    except Exception as e:
        st.warning(f"Erro ao buscar no Qdrant: {str(e)}")
        print(f"DEBUG: Exception in cache search: {e}")
        st.info("Tentando busca alternativa...")
        retrieved_docs = []

    # If no docs found, fetch from Spoonacular and store
    st.info("Buscando novas receitas no Spoonacular...")

    try:
        recipe_docs = fetch_recipes_by_ingredients(ingredients_list)

        if not recipe_docs:
            st.error(
                "Nenhuma receita encontrada com esses ingredientes na API do Spoonacular.")
            return None

        st.info(
            f"{len(recipe_docs)} receitas encontradas no Spoonacular. Armazenando no cache...")

        # Show debug info about Spoonacular results
        for i, doc in enumerate(recipe_docs[:2]):
            print(f"DEBUG: Spoonacular doc {i} metadata: {doc.metadata}")

        # Store in Qdrant for future use (this will also create the index)
        store_recipes(recipe_docs, config['embedder'], config['qdrant_url'],
                      config['qdrant_api_key'], config['collection_name'])

        st.success(
            "Receitas armazenadas com sucesso! Preparando índices para futuras buscas...")

        retrieved_docs = retrieve_recipes_by_ingredients(
            ingredients_list, config['embedder'], config['qdrant_url'],
            config['qdrant_api_key'], config['collection_name'], k=5
        )

        if not retrieved_docs:
            query = ", ".join(ingredients_list)
            retrieved_docs = retrieve_similar_recipes(
                query, config['embedder'], config['qdrant_url'],
                config['qdrant_api_key'], config['collection_name'], k=5
            )

        if retrieved_docs:
            st.success(
                f"{len(retrieved_docs)} receitas processadas com sucesso!")
            return retrieved_docs
        else:
            st.error("Não foi possível processar as receitas.")
            return None

    except Exception as e:
        st.error(f"Erro ao buscar receitas no Spoonacular: {str(e)}")
        print(f"DEBUG: Exception in Spoonacular search: {e}")
        return None


# def display_recipe_stats(retrieved_docs, user_ingredients):
#     """Display statistics about retrieved recipes"""
#     if not retrieved_docs:
#         return

#     st.subheader("Estatísticas das Receitas Encontradas")

#     col1, col2, col3 = st.columns(3)

#     with col1:
#         st.metric("Total de Receitas", len(retrieved_docs))

#     with col2:
#         # Count recipes that use all user ingredients
#         perfect_matches = 0
#         for doc in retrieved_docs:
#             used_ingredients = doc.metadata.get('used_ingredients', [])
#             if all(ing.lower() in [u.lower() for u in used_ingredients] for ing in user_ingredients):
#                 perfect_matches += 1
#         st.metric("Receitas com Todos Ingredientes", perfect_matches)

#     with col3:
#         # Average ingredient count
#         avg_ingredients = sum(doc.metadata.get('ingredient_count', 0)
#                               for doc in retrieved_docs) / len(retrieved_docs)
#         st.metric("Média de Ingredientes", f"{avg_ingredients:.1f}")

#     # Show some recipe titles
#     st.write("**Receitas Encontradas:**")
#     for i, doc in enumerate(retrieved_docs[:5], 1):
#         title = doc.metadata.get('title', 'Sem título')
#         used_ings = doc.metadata.get('used_ingredients', [])
#         st.write(
#             f"{i}. {title} (usa: {', '.join(used_ings[:3])}{'...' if len(used_ings) > 3 else ''})")

#     if len(retrieved_docs) > 5:
#         st.write(f"... e mais {len(retrieved_docs) - 5} receitas")


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

    st.title("Gerador de Receitas com RAG")
    st.markdown("*Powered by OpenAI + Qdrant + Spoonacular*")

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

            # Filter user ingredients to only include those found in the retrieved recipes
            # This removes ingredients that Spoonacular couldn't find recipes for
            valid_ingredients = filter_user_ingredients_by_recipes(
                ingredients_list, retrieved_docs)

            if not valid_ingredients:
                st.error(
                    "Nenhum dos ingredientes fornecidos foi encontrado em receitas. Tente ingredientes diferentes.")
                st.info(
                    "Sugestão: Use ingredientes comuns em inglês como 'chicken', 'tomato', 'cheese', 'egg', etc.")
                return

            # Show which ingredients are valid and which were removed
            if len(valid_ingredients) < len(ingredients_list):
                removed_ingredients = [
                    ing for ing in ingredients_list if ing not in valid_ingredients]
                st.warning(
                    f"Ingredientes removidos (não encontrados em receitas): {', '.join(removed_ingredients)}")
                st.success(
                    f"Ingredientes válidos que serão usados: {', '.join(valid_ingredients)}")
            else:
                st.success(
                    f"Todos os ingredientes são válidos: {', '.join(valid_ingredients)}")

            render_recipe_interface(
                valid_ingredients, context, config['openai_api_key'])


if __name__ == "__main__":
    main()
