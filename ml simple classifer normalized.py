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

    # VOTO
    voto_match = re.search(r'V\s*O\s*T\s*O(.*?)(?=(?:NEGA-SE|DECIDE-SE|É COMO VOTO|João Pessoa|Des\.))', text, re.IGNORECASE | re.DOTALL)
    if voto_match:
        result["voto"] = voto_match.group(1).strip()

    # Jurisprudências – pattern like: (REsp 123456/SP, Rel. Ministro Fulano)
    jurisprudencias = re.findall(r'\(\s*(?:REsp|AgRg|AgInt|AREsp)[^)]*\)', text)
    result["jurisprudencias"] = list(set(jurisprudencias))  # remove duplicates

    return result


# === MAIN ===

# Load JSONL (example: one entry per line)
with open("decisions.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        decision = json.loads(line)
        inteiro_teor = decision.get("inteiro_teor", "")
        extracted = extract_sections(inteiro_teor)

        print(f"Processo: {decision['numero']}")
        print("\n>> Acordão:\n", extracted["acordao"][:300], "...")  # print first 300 chars
        print("\n>> Relatório:\n", extracted["relatorio"][:300], "...")
        print("\n>> Voto:\n", extracted["voto"][:300], "...")
        print("\n>> Jurisprudências encontradas:")
        for j in extracted["jurisprudencias"]:
            print(" -", j)
        print("\n" + "="*80 + "\n")
