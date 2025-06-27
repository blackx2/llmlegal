## refined jurisprudence pipeline 1.2
## code with improvents, its use decision json from the webscraping project and use the decision outcome expressions from the judicial_results.json file

import pandas as pd
import re
import spacy
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

def classify_decision_outcome(text, positive_set, negative_set):
    text_lower = text.lower()
    for phrase in negative_set:
        if phrase.lower() in text_lower:
            return "negativo"
    for phrase in positive_set:
        if phrase.lower() in text_lower:
            return "positivo"
    return "indefinido"

def main():
    print("[+] Loading spaCy Portuguese model...")
    nlp = spacy.load("pt_core_news_md")

    # Load the scraped judicial decisions JSONL file
    print("[+] Loading data from decisions.jsonl ...")
    df = pd.read_json("decisions.jsonl", lines=True)

    # Load positive/negative expressions from external JSON
    print("[+] Loading decision outcome expressions from judicial_results.json ...")
    with open("judicial_results.json", encoding="utf-8") as f:
        outcome_data = json.load(f)
    positive_set = set(outcome_data["positive"])
    negative_set = set(outcome_data["negative"])

    # Classify decisions: positivo, negativo, indefinido
    print("[+] Classifying decision outcomes...")
    df["tipo_decisao"] = df["ementa"].apply(lambda x: classify_decision_outcome(x, positive_set, negative_set))
    print("\n[+] Distribution of decision outcomes:")
    print(df["tipo_decisao"].value_counts())

    # Preprocessing: clean, lowercase, lemmatize, remove stopwords/punct
    def preprocess_spacy(text):
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[\r\n]", " ", text)
        doc = nlp(text.lower())
        tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
        return " ".join(tokens).strip()

    print("[+] Preprocessing texts with spaCy...")
    df["ementa_clean"] = df["ementa"].apply(preprocess_spacy)
    df["inteiro_clean"] = df["inteiro_teor"].apply(preprocess_spacy)

    # Classification labels: negativo = 1, others = 0
    print("[+] Creating binary labels for classification...")
    df["resultado"] = df["tipo_decisao"].apply(lambda x: 1 if x == "negativo" else 0)

    print("[+] Vectorizing texts with TF-IDF...")
    tfidf = TfidfVectorizer(max_features=5000)
    X_tfidf = tfidf.fit_transform(df["ementa_clean"])

    print("[+] Training Random Forest classifier...")
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, df["resultado"], test_size=0.2, random_state=42)
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    print("\n[Classification Report]")
    print(classification_report(y_test, preds))

    # --- Clustering ---
    print("[+] Clustering decisions into 5 groups...")
    kmeans = KMeans(n_clusters=5, random_state=42)
    df["cluster"] = kmeans.fit_predict(X_tfidf)

    sns.countplot(data=df, x="cluster")
    plt.title("Decisions per Cluster")
    plt.xlabel("Cluster ID")
    plt.ylabel("Number of Decisions")
    plt.show()

    # --- Topic Modeling ---
    print("[+] Performing LDA Topic Modeling (5 topics)...")
    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    lda.fit(X_tfidf)
    terms = tfidf.get_feature_names_out()

    for i, topic in enumerate(lda.components_):
        print(f"\n[Topic {i+1}]")
        top_terms = [terms[i] for i in topic.argsort()[:-11:-1]]
        print(", ".join(top_terms))

    # --- Semantic Search ---
    print("[+] Loading SentenceTransformer multilingual model...")
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    print("[+] Generating embeddings for semantic search...")
    embeddings = model.encode(df["ementa_clean"].tolist(), show_progress_bar=True)

    query = "reajuste cassi"
    query_vec = model.encode([query])
    similarities = cosine_similarity(query_vec, embeddings)[0]
    df["similarity"] = similarities

    print("\n[Top 5 similar decisions to query: 'reajuste cassi']")
    top5 = df.sort_values("similarity", ascending=False)[["numero", "ementa"]].head(5)
    print(top5.to_string(index=False))


if __name__ == "__main__":
    main()
