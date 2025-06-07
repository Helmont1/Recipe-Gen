import os
import requests
from dotenv import load_dotenv
import json
from langchain.docstore.document import Document

load_dotenv()

SPOONACULAR_KEY = os.getenv("SPOONACULAR_API_KEY")


def fetch_recipes_by_ingredients(ingredients, num_recipes=10):
    """Fetch recipes from Spoonacular API and format as LangChain Documents"""
    url = "https://api.spoonacular.com/recipes/findByIngredients"
    params = {
        "ingredients": ",".join(ingredients),
        "number": num_recipes,
        "apiKey": SPOONACULAR_KEY,
        "instructionsRequired": True
    }

    response = requests.get(url, params=params)
    try:
        response.raise_for_status()
        recipes = response.json()
    except Exception as e:
        print(f"Error fetching recipes: {e}")
        print(f"Response content: {response.text}")
        return []

    docs = []
    for recipe in recipes:
        # Extract all ingredients (used + missed)
        all_ingredients = []

        # Used ingredients (ingredients from user input)
        used_ingredients = [ing['name'].lower()
                            for ing in recipe.get('usedIngredients', [])]
        all_ingredients.extend(used_ingredients)

        # Missed ingredients (additional ingredients in the recipe)
        missed_ingredients = [ing['name'].lower()
                              for ing in recipe.get('missedIngredients', [])]
        all_ingredients.extend(missed_ingredients)

        content = f"""
        Recipe Name: {recipe['title']}
        Ingredients: {json.dumps([ing['name'] for ing in recipe.get('usedIngredients', [])])}
        Instructions: {recipe.get('instructions', 'No instructions provided')}
        """

        # Enhanced metadata with ingredients for filtering
        metadata = {
            "id": recipe.get("id"),
            "title": recipe.get("title", ""),
            "ingredients": all_ingredients,  # List of ingredient names
            "used_ingredients": used_ingredients,  # User provided ingredients
            "missed_ingredients": missed_ingredients,  # Additional ingredients needed
            "ingredient_count": len(all_ingredients)
        }

        # Debug: Print metadata to see what's being stored
        print(f"DEBUG: Recipe '{recipe.get('title', 'Unknown')}' metadata:")
        print(f"  - ingredients: {all_ingredients}")
        print(f"  - used_ingredients: {used_ingredients}")
        print(f"  - missed_ingredients: {missed_ingredients}")

        docs.append(Document(page_content=content, metadata=metadata))

    return docs
