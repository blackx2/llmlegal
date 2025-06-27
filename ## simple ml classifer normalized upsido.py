## simple ml classifer normalized upsidown

import json
import re

def extract_sections(text):
    result = {
        "relatorio": None,
        "voto": None,
        "acordao": None,
        "jurisprudencias": []
    }

    # Normalize
    text = text.replace('\n', ' ').replace('\xa0', ' ')
    text = re.sub(r'\s{2,}', ' ', text).strip()

    # Try to isolate full acórdão
    acordao_match = re.search(r'(A\s*C\s*Ó\s*R\s*D\s*Ã\s*O.*?)R\s*E\s*L\s*A\s*T\s*Ó\s*R\s*I\s*O', text, re.IGNORECASE)
    if acordao_match:
        result["acordao"] = acordao_match.group(1).strip()

    # RELATÓRIO
    relatorio_match = re.search(r'R\s*E\s*L\s*A\s*T\s*Ó\s*R\s*I\s*O(.*?)(?=V\s*O\s*T\s*O)', text, re.IGNORECASE)
    if relatorio_match:
        result["relatorio"] = relatorio_match.group(1).strip()

    # VOTO — find all matches and take the last one (bottom to top)
    voto_matches = list(re.finditer(
        r'V\s*O\s*T\s*O(.*?)(?=(?:NEGA-SE|DECIDE-SE|É COMO VOTO|João Pessoa|Des\.|$))',
        text, re.IGNORECASE | re.DOTALL))
    if voto_matches:
        result["voto"] = voto_matches[-1].group(1).strip()

    # Jurisprudências
    jurisprudencias = re.findall(r'\s*(?:REsp|AgRg|AgInt|AREsp)[^)]*', text)
    result["jurisprudencias"] = list(set(jurisprudencias))

    return result

# === MAIN ===

with open("decisions.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        decision = json.loads(line)
        inteiro_teor = decision.get("inteiro_teor", "")
        extracted = extract_sections(inteiro_teor)

        print(f"Processo: {decision['numero']}")
        if extracted["acordao"]:
            print("\n>> Acórdão:\n", extracted["acordao"][:300], "...")
        if extracted["relatorio"]:
            print("\n>> Relatório:\n", extracted["relatorio"][:300], "...")
        if extracted["voto"]:
            print("\n>> Voto:\n", extracted["voto"][:300], "...")
        
        print("\n>> Jurisprudências encontradas:")
        for j in extracted["jurisprudencias"]:
            print(" -", j)
        print("\n" + "="*80 + "\n")