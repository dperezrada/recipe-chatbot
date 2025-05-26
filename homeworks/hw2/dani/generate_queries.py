import os
import pandas as pd
from utils import call_llm
from backend.utils import get_agent_response

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

dimensions = """
### Dimension 1: DietaryNeedsOrRestrictions
- **vegan** (no animal products)
- **vegetarian** (no meat, but includes dairy and eggs)
- **gluten-free** (no wheat, barley, rye)
- **dairy-free** (no milk, cheese, butter)
- **keto** (high fat, very low carbohydrate)
- **paleo** (whole foods, no processed)
- **halal** (Islamic dietary laws)
- **kosher** (Jewish dietary laws)
- **no restrictions** (any ingredients)
- **pescatarian** (fish but no meat)
- **low-carb** (reduced carbohydrates)
- **low-sodium** (reduced salt)
- **nut-free** (no tree nuts or peanuts)
- **egg-free** (no eggs)
- **soy-free** (no soy products)
- **FODMAP** (low fermentable carbs)
- **diabetic-friendly** (blood sugar conscious)
- **high-protein** (protein-focused meals)

### Dimension 2: AvailableIngredientsFocus
- **must_use_specific** (specific ingredients that must be included)
- **general_pantry** (basic pantry ingredients)
- **no_specific_ingredients** (open to any suggestions)

### Dimension 3: CuisinePreference
- **Italian** (pasta, pizza, risotto)
- **Asian** (Chinese, Japanese, Thai, Korean)
- **Mexican** (tacos, enchiladas, quesadillas)
- **Mediterranean** (Greek, Middle Eastern)
- **American** (BBQ, comfort food, Southern)
- **Indian** (curry, spices, regional dishes)
- **French** (classic techniques, sauces)
- **any_cuisine** (no preference)
- **avoid_specific** (excluding certain cuisines)

### Dimension 4: SkillLevelEffort
- **beginner_easy_low_effort** (simple techniques, minimal prep)
- **intermediate_moderate_effort** (some cooking skills required)
- **advanced_complex_high_effort** (complex techniques, time-intensive)

### Dimension 5: TimeAvailability
- **quick_under_30_mins** (fast meals, minimal cooking time)
- **moderate_30_to_60_mins** (standard cooking time)
- **flexible_no_time_constraint** (can take as long as needed)

### Dimension 6: QueryStyleAndDetail
- **short_keywords_minimal_detail** (brief, keyword-based queries)
- **natural_question_moderate_detail** (conversational questions)
- **detailed_request_high_detail** (comprehensive, specific requests)
"""

prompt_tuples = f"""
Generate 10 unique combinations of the following dimensions:

{dimensions}
# OUTPUT FORMAT
(Dimension 1, Dimension 2, Dimension 3, Dimension 4, Dimension 5, Dimension 6)
(Dimension 1, Dimension 2, Dimension 3, Dimension 4, Dimension 5, Dimension 6)
"""

def generate_queries(tuples: list[str], n=10):
    prompt_queries = f"""
    Generate {n} unique queries for the following tuples

    # INSTRUCTIONS
    The queries should:
    1. Sound like real users asking for recipe help
    2. Naturally incorporate all the dimension values
    3. Vary in style and detail level
    4. Include natural variations like lowercase, typos, missing punctuation, emojis
  
    # INPUT FORMAT
    {tuples}

    # OUTPUT FORMAT
    Query\tTuple
    Query\tTuple
    """
    messages = [
        {"role": "system", "content": prompt_queries},
    ]
    response = call_llm(messages)
    return response

def generate_tuples():
    messages = [
        {"role": "system", "content": prompt_tuples},
    ]
    response = call_llm(messages)
    return response

def generate_query_response(query: str):
    messages = [
        {"role": "user", "content": query},
    ]
    response = get_agent_response(messages)
    return response[-1]["content"].replace("```markdown", "").replace("```", "").replace("\t", "").replace('"', '')

def main():
    tuples = generate_tuples()
    queries = generate_queries(tuples, n=20)
    rows = []
    for index, line in enumerate(queries.split("\n")):
        if index == 0:
            continue
        if line.strip() == "":
            continue
        print(line)
        try:
            query, tuple = line.split("\t")
        except Exception as e:
            print(f"Error splitting line: {line}")
            continue
        response = generate_query_response(query)
        rows.append({"query": query, "tuple": tuple, "response": response})
    # Convert to DataFrame
    df = pd.DataFrame(rows)
    
    # Save to CSV
    df.to_csv(f"{CURRENT_DIR}/queries.tsv", index=False)

if __name__ == "__main__":
    main()
