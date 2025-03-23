import streamlit as st
from src.rag.pipeline import generate_recipe
import asyncio
import os

os.environ["LANGCHAIN_ALLOW_DANGEROUS_DESERIALIZATION"] = "true"


try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

st.title("AI Recipe Generator 🍳")
ingredients = st.text_input("Enter ingredients (comma-separated):")
if ingredients:
    ingredients_list = [x.strip() for x in ingredients.split(",")]
    recipe = generate_recipe(ingredients_list)
    st.write("### Your Custom Recipe")
    st.write(recipe)