import json
import re
from collections import Counter

# Step 1 – Load decisions
with open("decisions.jsonl", "r", encoding="utf-8") as f:
    decisions = [json.loads(line) for line in f]

# Step 2 – Load expressions from judicial_results.json
with open("judicial_results.json", "r", encoding="utf-8") as f:
    expressions = json.load(f)

# Step 3 – Compile regex patterns for both sets
def compile_patterns(expression_list):
    return [re.compile(re.escape(expr.strip()), re.IGNORECASE) for expr in expression_list]

positive_patterns = compile_patterns(expressions["positive"])
negative_patterns = compile_patterns(expressions["negative"])

# Step 4 – Classify the text of each decision
def classify_text(text):
    pos_found = any(p.search(text) for p in positive_patterns)
    neg_found = any(n.search(text) for n in negative_patterns)

    if pos_found and neg_found:
        return "parcialmente_favoravel"
    elif pos_found:
        return "favoravel"
    elif neg_found:
        return "nao_favoravel"
    else:
        return "indefinido"

# Step 5 – Apply classification
results = []
for dec in decisions:
    combined_text = f"{dec.get('ementa', '')} {dec.get('inteiro_teor', '')}".lower()
    label = classify_text(combined_text)
    dec["classificacao"] = label
    results.append(dec)

# Step 6 – Save results with classification
with open("classified_decisions.jsonl", "w", encoding="utf-8") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

# Step 7 – Print summary
summary = Counter(d["classificacao"] for d in results)
print("\n📊 Classificação dos casos:")
for categoria, total in summary.items():
    print(f"{categoria}: {total}")

# Step 8 – Print detailed results
print("\n📄 Processos e suas classificações:")
for dec in results:
    print(f"{dec['numero']}: {dec['classificacao']}")
