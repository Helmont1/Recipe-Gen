from langchain_openai import ChatOpenAI
import json


def generate_recipe(context, ingredients, openai_api_key):
    context_prompt = f"""
    Usando apenas estes ingredientes: {', '.join(ingredients)}, e considerando as seguintes receitas como inspiração: {context}, crie uma nova receita. Não adicione outros ingredientes, exceto temperos, ervas ou especiarias que possam realçar o sabor, caso faça sentido. Inclua quantidades (em gramas) e instruções passo a passo. Se faltar algum ingrediente, mencione nas instruções, mas não invente ou adicione novos.
    Seja claro e detalhista no passo a passo.
    Retorne a resposta como JSON estruturado:
    {{
        "title": "...",
        "ingredients": ["ingrediente1", "ingrediente2", ...],
        "steps": ["passo1", "passo2", ...],
        "nutrition_markdown": "| Nutriente | Valor |\n|---|---|\n| Calorias | ... | ... | ... | ... |",
    }}
    A receita e todos os campos devem estar em português. A informação nutricional deve ser uma tabela markdown no campo 'nutrition_markdown'.
    """
    llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        model="gpt-4.1-nano",
        temperature=0.7,
        max_tokens=1024
    )
    final_recipe = llm.invoke(context_prompt)
    try:
        recipe_json = json.loads(final_recipe.content if hasattr(
            final_recipe, 'content') else final_recipe)
        return recipe_json
    except Exception:
        return final_recipe.content if hasattr(final_recipe, 'content') else final_recipe
