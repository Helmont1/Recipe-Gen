import os
import requests
from dotenv import load_dotenv

load_dotenv()

SPOONACULAR_KEY = os.getenv("SPOONACULAR_API_KEY")

def fetch_recipes_by_ingredients(ingredients: list, number=5):
    """Fetch recipes from Spoonacular API using ingredients."""
    url = "https://api.spoonacular.com/recipes/findByIngredients"
    params = {
        "ingredients": ",".join(ingredients),
        "number": number,
        "apiKey": SPOONACULAR_KEY
    }
    response = requests.get(url, params=params)
    return response.json()