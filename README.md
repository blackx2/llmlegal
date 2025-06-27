# Jurimetrics Pipeline for Brazilian Judicial Decisions

An end-to-end Natural Language Processing (NLP) pipeline designed to analyze and extract insights from Brazilian judicial decisions. This project combines text preprocessing, classification, clustering, topic modeling, and semantic search to support jurimetrics and legal analytics research.

## Features

- **Text Preprocessing:**  
  Clean and normalize Portuguese legal texts using spaCy with lemmatization, stopword removal, and punctuation filtering.

- **Classification:**  
  Binary classification using TF-IDF vectorization and Random Forest to identify decisions labeled as "improcedente".

- **Clustering:**  
  KMeans clustering to group judicial decisions into meaningful clusters.

- **Topic Modeling:**  
  Latent Dirichlet Allocation (LDA) for extracting dominant topics within legal texts.

- **Semantic Search:**  
  Sentence Transformers for generating embeddings and performing semantic similarity searches across judicial summaries.

## Technologies

- Python 3  
- pandas  
- spaCy (`pt_core_news_md`)  
- scikit-learn  
- sentence-transformers  
- matplotlib & seaborn
