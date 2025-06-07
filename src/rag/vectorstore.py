from langchain.vectorstores import Qdrant
from langchain_qdrant import QdrantVectorStore
from qdrant_client.models import Filter, FieldCondition, MatchAny, PayloadSchemaType
from qdrant_client import QdrantClient


def get_vectorstore(embedder, url, api_key, collection_name):
    return QdrantVectorStore.from_existing_collection(
        embedding=embedder,
        collection_name=collection_name,
        url=url,
        api_key=api_key
    )


def create_ingredients_index(url, api_key, collection_name):
    """Create index for ingredients field to enable filtering"""
    try:
        client = QdrantClient(url=url, api_key=api_key)

        # Check if collection exists
        collections = client.get_collections()
        collection_exists = any(
            col.name == collection_name for col in collections.collections)

        if collection_exists:
            # Create index for ingredients field
            client.create_payload_index(
                collection_name=collection_name,
                field_name="used_ingredients",
                field_schema=PayloadSchemaType.KEYWORD
            )
            print(
                f"Index created for 'ingredients' field in collection '{collection_name}'")
            return True
    except Exception as e:
        print(f"Error creating index: {e}")
        return False


def store_recipes(recipes, embedder, url, api_key, collection_name):
    """Store recipes with enhanced metadata and create necessary indexes"""
    # Store documents first
    Qdrant.from_documents(
        recipes,
        embedder,
        url=url,
        api_key=api_key,
        collection_name=collection_name
    )

    # Create index for ingredients field after storing documents
    create_ingredients_index(url, api_key, collection_name)


def create_ingredients_filter(ingredients_list):
    """Create Qdrant filter for recipes containing any of the specified ingredients"""
    if not ingredients_list:
        print("DEBUG: No ingredients provided for filter")
        return None

    # Normalize ingredients to lowercase
    normalized_ingredients = [ing.lower().strip() for ing in ingredients_list]
    print(
        f"DEBUG: Normalized ingredients for filter: {normalized_ingredients}")

    try:
        # Create filter to match recipes that contain any of the user ingredients
        ingredients_filter = Filter(
            must=[
                FieldCondition(
                    key="used_ingredients",
                    match=MatchAny(any=normalized_ingredients)
                )
            ]
        )
        print(f"DEBUG: Filter created successfully")
        return ingredients_filter
    except Exception as e:
        print(f"DEBUG: Error creating filter: {e}")
        return None


def retrieve_similar_recipes(query, embedder, url, api_key, collection_name, k=5, ingredients_filter=None):
    """Retrieve similar recipes with optional ingredient filtering"""
    vector_db = get_vectorstore(embedder, url, api_key, collection_name)

    if ingredients_filter:
        try:
            # Use filtered similarity search
            return vector_db.similarity_search(
                query,
                k=k,
                filter=ingredients_filter
            )
        except Exception as e:
            print(f"Filtered search failed: {e}")
            # Fallback to regular search if filtering fails
            return vector_db.similarity_search(query, k=k)
    else:
        # Standard similarity search
        return vector_db.similarity_search(query, k=k)


def retrieve_recipes_by_ingredients_fallback(ingredients_list, embedder, url, api_key, collection_name, k=10):
    """Fallback method: retrieve all recipes and filter in memory"""
    try:
        vector_db = get_vectorstore(embedder, url, api_key, collection_name)

        # Get more documents to filter locally
        query = f"receita com {', '.join(ingredients_list)}"
        all_docs = vector_db.similarity_search(
            query, k=k*3)  # Get 3x more to filter
        print(f"DEBUG: Fallback retrieved {len(all_docs)} docs for filtering")

        # Filter in memory by checking metadata
        filtered_docs = []
        normalized_user_ingredients = [
            ing.lower().strip() for ing in ingredients_list]
        print(
            f"DEBUG: Filtering for ingredients: {normalized_user_ingredients}")

        for i, doc in enumerate(all_docs):
            used_ingredients = doc.metadata.get('used_ingredients', [])
            print(f"DEBUG: Doc {i} used_ingredients: {used_ingredients}")

            # Check if any user ingredient is contained in any recipe ingredient
            matches = []
            for user_ing in normalized_user_ingredients:
                for recipe_ing in used_ingredients:
                    if user_ing in recipe_ing.lower():
                        matches.append(f"{user_ing} -> {recipe_ing}")
                        break  # Found match for this user ingredient

            if matches:
                print(f"DEBUG: Doc {i} matches: {matches}")
                filtered_docs.append(doc)

        print(f"DEBUG: Fallback filtered to {len(filtered_docs)} docs")
        return filtered_docs[:k]  # Return only requested number

    except Exception as e:
        print(f"Fallback search failed: {e}")
        return []


def retrieve_recipes_by_ingredients(ingredients_list, embedder, url, api_key, collection_name, k=10):
    """Retrieve recipes specifically filtered by ingredients with fallback"""

    # Since Qdrant filtering requires exact matches and our ingredients are complex strings,
    # we'll skip the Qdrant filter and use the fallback method directly
    print(f"DEBUG: Using fallback method directly due to complex ingredient names")

    return retrieve_recipes_by_ingredients_fallback(
        ingredients_list, embedder, url, api_key, collection_name, k
    )
