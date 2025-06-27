## jurimetrics_pipeline_brazil.py

import pandas as pd
import re
import spacy
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

def classify_decision_outcome(text):
    text_lower = text.lower()
    if "parcialmente procedente" in text_lower:
        return "parcialmente procedente"
    elif "procedente" in text_lower or "acolhido" in text_lower:
        return "procedente"
    elif "improcedente" in text_lower or "rejeitado" in text_lower or "não acolhido" in text_lower:
        return "improcedente"
    else:
        return "indefinido"

def main():
    print("[+] Loading spaCy Portuguese model...")
    nlp = spacy.load("pt_core_news_md")

    # Load the scraped judicial decisions JSONL file
    print("[+] Loading data from decisions.jsonl ...")
    df = pd.read_json("decisions.jsonl", lines=True)

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

    # --- NEW: Classify decision outcome from original 'ementa' text ---
    print("[+] Classifying decision outcomes...")
    df["tipo_decisao"] = df["ementa"].apply(classify_decision_outcome)
    print("\n[+] Distribution of decision outcomes:")
    print(df["tipo_decisao"].value_counts())

    # --- Step 1: Classification (dummy binary label: improcedente = 1, else 0) ---
    print("[+] Creating dummy labels for classification...")
    df["resultado"] = df["tipo_decisao"].apply(lambda x: 1 if x == "improcedente" else 0)

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

    # --- Step 2: Clustering ---
    print("[+] Clustering decisions into 5 groups...")
    kmeans = KMeans(n_clusters=5, random_state=42)
    df["cluster"] = kmeans.fit_predict(X_tfidf)

    sns.countplot(data=df, x="cluster")
    plt.title("Decisions per Cluster")
    plt.xlabel("Cluster ID")
    plt.ylabel("Number of Decisions")
    plt.show()

    # --- Step 3: Topic Modeling ---
    print("[+] Performing LDA Topic Modeling (5 topics)...")
    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    lda.fit(X_tfidf)
    terms = tfidf.get_feature_names_out()

    for i, topic in enumerate(lda.components_):
        print(f"\n[Topic {i+1}]")
        top_terms = [terms[i] for i in topic.argsort()[:-11:-1]]
        print(", ".join(top_terms))

    # --- Step 4: Semantic Search with Sentence Transformers ---
    print("[+] Loading SentenceTransformer multilingual model...")
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    print("[+] Generating embeddings for semantic search...")
    embeddings = model.encode(df["ementa_clean"].tolist(), show_progress_bar=True)

    # Example search query
    query = "reajuste cassi"
    query_vec = model.encode([query])
    similarities = cosine_similarity(query_vec, embeddings)[0]
    df["similarity"] = similarities

    print("\n[Top 5 similar decisions to query: 'reajuste cassi']")
    top5 = df.sort_values("similarity", ascending=False)[["numero", "ementa"]].head(5)
    print(top5.to_string(index=False))


if __name__ == "__main__":
    main()
