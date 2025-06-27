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
from transformers import pipeline
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print("[+] Loading spaCy Portuguese model...")
    nlp = spacy.load("pt_core_news_md")

    print("[+] Loading data from decisions.jsonl ...")
    df = pd.read_json("decisions.jsonl", lines=True)
    print(f"[+] Total decisions loaded: {len(df)}")

    # --- AI-Based Classification with Language Model ---
    print("[+] Loading multilingual zero-shot classifier...")
    classifier = pipeline("zero-shot-classification", model="joeddav/xlm-roberta-large-xnli")

    candidate_labels = ["favorável", "desfavorável"]

    def classify_ai(text):
        result = classifier(text, candidate_labels)
        top_label = result["labels"][0]
        score = result["scores"][0]
        return top_label if score >= 0.6 else "indefinido"

    print("[+] Classifying decisions with AI model...")
    df["tipo_decisao"] = df["ementa"].apply(classify_ai)
    print("\n[+] Distribution of AI-predicted outcomes:")
    print(df["tipo_decisao"].value_counts())

    # Preprocess text
    def preprocess_spacy(text):
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[\r\n]", " ", text)
        doc = nlp(text.lower())
        tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
        return " ".join(tokens).strip()

    print("[+] Preprocessing texts with spaCy...")
    df["ementa_clean"] = df["ementa"].apply(preprocess_spacy)
    df["inteiro_clean"] = df["inteiro_teor"].apply(preprocess_spacy)

    print("[+] Creating binary label: desfavorável = 1, others = 0")
    df["resultado"] = df["tipo_decisao"].apply(lambda x: 1 if x == "desfavorável" else 0)

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

    print("[+] Clustering decisions into 5 groups...")
    kmeans = KMeans(n_clusters=5, random_state=42)
    df["cluster"] = kmeans.fit_predict(X_tfidf)

    sns.countplot(data=df, x="cluster")
    plt.title("Decisions per Cluster")
    plt.xlabel("Cluster ID")
    plt.ylabel("Number of Decisions")
    plt.show()

    print("[+] Performing LDA Topic Modeling (5 topics)...")
    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    lda.fit(X_tfidf)
    terms = tfidf.get_feature_names_out()

    for i, topic in enumerate(lda.components_):
        print(f"\n[Topic {i+1}]")
        top_terms = [terms[i] for i in topic.argsort()[:-11:-1]]
        print(", ".join(top_terms))

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
