from langchain_openai import ChatOpenAI
import json


def generate_recipe(context, ingredients, openai_api_key, generated_recipes=[]):
    context_prompt = f"""
    Usando unica e exclusivamente apenas estes ingredientes: {', '.join(ingredients)}, e considerando as seguintes receitas como inspiração: {context}, crie uma nova receita. Não adicione outros ingredientes, exceto temperos, ervas ou especiarias que possam realçar o sabor, caso faça sentido. Inclua quantidades (em gramas) e instruções passo a passo. Se faltar algum ingrediente, mencione nas instruções, mas não invente ou adicione novos.
    Seja claro e detalhista no passo a passo.
    Retorne a resposta como JSON estruturado:
    {{
        "title": "...",
        "ingredients": ["ingrediente1", "ingrediente2", ...],
        "steps": ["passo1", "passo2", ...],
        "nutrition_markdown": "| Nutriente | Valor |\n|---|---|\n| Calorias | ... | ... | ... | ... |",
    }}
    A receita e todos os campos devem estar em português. A informação nutricional deve ser uma tabela markdown no campo 'nutrition_markdown', e deve conter a quantidade total de calorias, proteinas, carboidratos, gorduras, fibras e sódio, dos ingredientes da receita, e na quantidade total de cada ingrediente.
    """
    if len(generated_recipes) > 0:
        context_prompt += f"""
        Se possível, a receita deve ser diferente das receitas abaixo:
        {', '.join(generated_recipes)}
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
