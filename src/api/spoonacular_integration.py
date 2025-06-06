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
        return []  # Return empty list on error

    docs = []
    for recipe in recipes:
        content = f"""
        Recipe Name: {recipe['title']}
        Ingredients: {json.dumps([ing['name'] for ing in recipe['usedIngredients']])}
        Instructions: {recipe.get('instructions', 'No instructions provided')}
        """
        docs.append(Document(page_content=content,
                    metadata={"id": recipe["id"]}))

    return docs
