def parse_ingredients(ingredients_str):
    return [x.strip() for x in ingredients_str.split(",") if x.strip()]
