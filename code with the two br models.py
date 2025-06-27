import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

# Define model names
model_names = {
    "legalbert": "dominguesm/legal-bert-base-cased-ptbr",
    "legalbert_fp": "raquelsilveira/legalbertpt_fp"
}

# Load models and tokenizers
tokenizers = {name: AutoTokenizer.from_pretrained(model_id) for name, model_id in model_names.items()}
models = {name: AutoModel.from_pretrained(model_id).eval() for name, model_id in model_names.items()}

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for model in models.values():
    model.to(device)

# Few-shot example paragraphs for each section
section_examples = {
    "relatorio": [
        "O relatório apresenta os fatos relevantes do processo conforme a narrativa das partes e os documentos anexados.",
        "Relatório. Trata-se de recurso interposto por..."
    ],
    "voto": [
        "É como voto.",
        "Diante do exposto, voto no sentido de negar provimento ao recurso."
    ],
    "acordao": [
        "A C O R D Ã O. Vistos, relatados e discutidos os autos, decide o Tribunal...",
        "Decide-se, portanto, que a decisão está de acordo com a jurisprudência consolidada."
    ],
    "jurisprudencia": [
        "Precedente: (REsp 123456/SP, Rel. Min. João, DJe 01/01/2020)",
        "Conforme entendimento do STJ: (AgRg no AREsp 78910/PR)..."
    ]
}

# Build average embedding for each label using examples
reference_embeddings = {}
for label, examples in section_examples.items():
    reference_embeddings[label] = {}
    for name in model_names:
        embs = []
        for example in examples:
            encoded = tokenizers[name](example, return_tensors="pt", truncation=True, max_length=512).to(device)
            with torch.no_grad():
                output = models[name](**encoded)
                emb = output.last_hidden_state[0][0]
                embs.append(emb)
        # Average the embeddings
        reference_embeddings[label][name] = torch.stack(embs).mean(dim=0)

# Function to compute CLS embedding
def get_cls_embedding(text, model, tokenizer):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        outputs = model(**inputs)
        return outputs.last_hidden_state[0][0]

# Classify a chunk of text
def classify_chunk(text):
    scores = {label: 0.0 for label in section_examples.keys()}
    for name in model_names:
        emb = get_cls_embedding(text, models[name], tokenizers[name])
        for label in section_examples.keys():
            sim = F.cosine_similarity(emb, reference_embeddings[label][name], dim=0)
            scores[label] += sim.item()
    return max(scores, key=scores.get)

# === MAIN ===
with open("decisions.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        decision = json.loads(line)
        inteiro_teor = decision.get("inteiro_teor", "")
        print(f"\n📄 Processo: {decision['numero']}")

        # Split into paragraphs (filter out very short ones)
        paragraphs = [p.strip() for p in inteiro_teor.split('\n') if len(p.strip()) > 100]

        # Classify and group paragraphs
        classified = {label: [] for label in section_examples.keys()}
        for p in paragraphs:
            label = classify_chunk(p)
            classified[label].append(p)

        # Print each section
        for label in ["acordao", "relatorio", "voto"]:
            print(f"\n📘 {label.upper()}:\n")
            print(" ".join(classified[label])[:500], "...\n")

        print("\n⚖️ JURISPRUDÊNCIAS:\n")
        for j in classified["jurisprudencia"]:
            print(" -", j[:300].replace("\n", " ") + "...")

        print("\n" + "=" * 80 + "\n")
