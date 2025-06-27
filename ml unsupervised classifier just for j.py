## ml unsupervised classifier just for jurisprudence
import re
import json
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN

# Load Sentence-BERT for Portuguese or multilingual
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def extract_candidate_citations(text):
    # Broader regex: match anything in parentheses up to 100 chars that has legal abbreviations or digits
    pattern = r'\(([^)]{5,100}?)\)'
    candidates = re.findall(pattern, text)

    # Filter candidates by presence of typical jurisprudence keywords or digits
    keywords = ['REsp', 'AgRg', 'AgInt', 'AREsp', 'RE', 'STF', 'TJSP', 'STJ']
    filtered = [c for c in candidates if any(k.lower() in c.lower() for k in keywords) or re.search(r'\d{4,}', c)]

    return filtered

def cluster_citations(citations):
    if not citations:
        return []

    embeddings = model.encode(citations)
    # DBSCAN groups nearby citations; eps and min_samples can be tuned
    clustering = DBSCAN(eps=0.3, min_samples=2, metric='cosine').fit(embeddings)

    clusters = {}
    for idx, label in enumerate(clustering.labels_):
        if label == -1:
            # Noise, keep as separate
            clusters.setdefault('noise', []).append(citations[idx])
        else:
            clusters.setdefault(label, []).append(citations[idx])

    # Return all citations in clusters except noise (or keep noise if you want)
    grouped_citations = []
    for k, group in clusters.items():
        grouped_citations.extend(group)

    return list(set(grouped_citations))

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

    # Your existing regex for acordao, relatorio, voto
    acordao_match = re.search(r'(A\s*C\s*Ó\s*R\s*D\s*Ã\s*O.*?)R\s*E\s*L\s*A\s*T\s*Ó\s*R\s*I\s*O', text, re.IGNORECASE)
    if acordao_match:
        result["acordao"] = acordao_match.group(1).strip()

    relatorio_match = re.search(r'R\s*E\s*L\s*A\s*T\s*Ó\s*R\s*I\s*O(.*?)(?=V\s*O\s*T\s*O)', text, re.IGNORECASE)
    if relatorio_match:
        result["relatorio"] = relatorio_match.group(1).strip()

    voto_match = re.findall(r'V\s*O\s*T\s*O(.*?)(?=(?:NEGA-SE|DECIDE-SE|É COMO VOTO|João Pessoa|Des\.))', text, re.IGNORECASE | re.DOTALL)
    if voto_match:
        result["voto"] = [match.strip() for match in voto_match]

    # Improved jurisprudencias detection
    candidates = extract_candidate_citations(text)
    clustered_citations = cluster_citations(candidates)
    result["jurisprudencias"] = clustered_citations

    return result

# Example usage:
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
        
# Write extracted data to JSONL
output_file = "extracted_sections.jsonl"

with open("decisions.jsonl", "r", encoding="utf-8") as f_in, open(output_file, "w", encoding="utf-8") as f_out:
    for line in f_in:
        decision = json.loads(line)
        inteiro_teor = decision.get("inteiro_teor", "")
        extracted = extract_sections(inteiro_teor)

        structured = {
            "numero": decision.get("numero"),
            "acordao": extracted["acordao"],
            "relatorio": extracted["relatorio"],
            "voto": extracted["voto"],
            "jurisprudencias": extracted["jurisprudencias"]
        }

        # Save as JSON line
        f_out.write(json.dumps(structured, ensure_ascii=False) + "\n")

        # Optional: print summary
        print(f"Processo: {structured['numero']}")
        if structured["acordao"]:
            print("\n>> Acórdão:\n", structured["acordao"][:300], "...")
        if structured["relatorio"]:
            print("\n>> Relatório:\n", structured["relatorio"][:300], "...")
        if structured["voto"]:
            print("\n>> Voto:\n", structured["voto"][0][:300], "...")  # show first if multiple

        print("\n>> Jurisprudências encontradas:")
        for j in structured["jurisprudencias"]:
            print(" -", j)
        print("\n" + "="*80 + "\n")