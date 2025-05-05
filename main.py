import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
import string
import re
import os
import warnings
from wordcloud import WordCloud
from collections import Counter
import matplotlib.colors as mcolors
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
from scipy.sparse import csr_matrix
import math
import time
import pickle
import logging

# Konfigurasi logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download NLTK resources
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    logger.info("NLTK resources downloaded successfully")
except Exception as e:
    logger.error(f"Error downloading NLTK resources: {e}")

class TFIDFAnalyzer:
    """
    Kelas untuk menganalisis dokumen menggunakan TF-IDF dengan visualisasi kompleks.
    """
    
    def __init__(self, language='english', custom_stopwords=None, min_df=2, max_df=0.95, 
                 ngram_range=(1, 2), use_lemmatization=True, use_stemming=False):
        """
        Inisialisasi TFIDFAnalyzer.
        
        Parameters:
        -----------
        language : str, default='english'
            Bahasa yang digunakan untuk stopwords dan lemmatization
        custom_stopwords : list, default=None
            Daftar stopwords tambahan
        min_df : int or float, default=2
            Frekuensi dokumen minimum untuk kata yang dipertimbangkan
        max_df : float, default=0.95
            Frekuensi dokumen maksimum untuk kata yang dipertimbangkan
        ngram_range : tuple, default=(1, 2)
            Rentang n-gram (min_n, max_n)
        use_lemmatization : bool, default=True
            Menggunakan lemmatization jika True
        use_stemming : bool, default=False
            Menggunakan stemming jika True
        """
        self.language = language
        self.min_df = min_df
        self.max_df = max_df
        self.ngram_range = ngram_range
        self.use_lemmatization = use_lemmatization
        self.use_stemming = use_stemming
        
        # Siapkan stopwords
        self.stop_words = set(stopwords.words(language))
        if custom_stopwords:
            self.stop_words.update(custom_stopwords)
        
        # Inisialisasi alat NLP
        self.stemmer = PorterStemmer() if use_stemming else None
        self.lemmatizer = WordNetLemmatizer() if use_lemmatization else None
        
        # Inisialisasi vectorizer
        # self.tfidf_vectorizer = TfidfVectorizer(
        #     min_df=min_df,
        #     max_df=max_df,
        #     ngram_range=ngram_range,
        #     stop_words=None,  # Kita akan melakukan preprocessing sendiri
        #     tokenizer=None,  # Kita akan melakukan tokenization sendiri
        #     preprocessor=None,  # Kita akan melakukan preprocessing sendiri
        #     token_pattern=None,  # Kita akan melakukan tokenization sendiri
        # )
        
        self.tfidf_vectorizer = TfidfVectorizer(
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram_range,
            stop_words=None,  # Already handled in preprocessing
            tokenizer=lambda text: text.split(),  # Split preprocessed text by spaces
            preprocessor=None,  # Already handled in preprocessing
            token_pattern=None,  # Use custom tokenizer instead of regex
        )
        
        # Untuk menyimpan hasil
        self.documents = None
        self.processed_documents = None
        self.tfidf_matrix = None
        self.feature_names = None
        self.document_names = None
        self.document_clusters = None
        self.top_terms = None
        self.execution_time = {}
        
    def preprocess_text(self, text):
        """
        Melakukan preprocessing pada teks.
        
        Parameters:
        -----------
        text : str
            Teks yang akan diproses
            
        Returns:
        --------
        str
            Teks yang telah diproses
        """
        # Konversi ke lowercase
        text = text.lower()
        
        # Menghapus URL
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Menghapus tag HTML
        text = re.sub(r'<.*?>', '', text)
        
        # Menghapus angka
        text = re.sub(r'\d+', '', text)
        
        # Menghapus tanda baca
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Tokenisasi
        tokens = word_tokenize(text)
        
        # Menghapus stopwords
        tokens = [token for token in tokens if token not in self.stop_words]
        
        # Stemming/lemmatization
        if self.use_stemming:
            tokens = [self.stemmer.stem(token) for token in tokens]
        elif self.use_lemmatization:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        # Menggabungkan kembali token-token
        processed_text = ' '.join(tokens)
        
        return processed_text
    
    def fit(self, documents, document_names=None):
        """
        Melatih model TF-IDF pada dokumen-dokumen.
        
        Parameters:
        -----------
        documents : list
            Daftar dokumen teks
        document_names : list, default=None
            Daftar nama dokumen
        
        Returns:
        --------
        self
        """
        start_time = time.time()
        
        self.documents = documents
        if document_names:
            self.document_names = document_names
        else:
            self.document_names = [f"Doc {i+1}" for i in range(len(documents))]
        
        logger.info("Preprocessing documents...")
        self.processed_documents = [self.preprocess_text(doc) for doc in documents]
        
        logger.info("Calculating TF-IDF matrix...")
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.processed_documents)
        self.feature_names = np.array(self.tfidf_vectorizer.get_feature_names_out())
        
        self.execution_time['fit'] = time.time() - start_time
        logger.info(f"TF-IDF matrix created with shape {self.tfidf_matrix.shape}")
        
        return self
    
    def calculate_manual_tfidf(self):
        """
        Menghitung TF-IDF secara manual untuk perbandingan dan pembelajaran.
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame berisi nilai TF-IDF untuk setiap term dan dokumen
        """
        start_time = time.time()
        
        # Menghitung kemunculan term dalam setiap dokumen
        count_vectorizer = CountVectorizer(
            min_df=self.min_df,
            max_df=self.max_df,
            ngram_range=self.ngram_range,
        )
        count_matrix = count_vectorizer.fit_transform(self.processed_documents)
        terms = count_vectorizer.get_feature_names_out()
        
        # Membuat DataFrame untuk hasil TF-IDF
        manual_tfidf = pd.DataFrame(
            0, 
            index=self.document_names, 
            columns=terms
        )
        
        # Menghitung TF-IDF secara manual
        N = len(self.processed_documents)  # Jumlah dokumen
        
        # Untuk setiap dokumen
        for doc_idx, doc_name in enumerate(self.document_names):
            # Untuk setiap term
            for term_idx, term in enumerate(terms):
                # Term frequency dalam dokumen ini
                tf = count_matrix[doc_idx, term_idx]
                
                # Jumlah dokumen yang mengandung term ini
                df = np.sum(count_matrix.toarray()[:, term_idx] > 0)
                
                # Menghitung IDF
                idf = np.log((N + 1) / (df + 1)) + 1  # Smoothing
                
                # Menghitung TF-IDF
                tfidf = tf * idf
                manual_tfidf.loc[doc_name, term] = tfidf
        
        self.execution_time['manual_tfidf'] = time.time() - start_time
        logger.info(f"Manual TF-IDF calculation completed in {self.execution_time['manual_tfidf']:.2f} seconds")
        
        return manual_tfidf
    
    def get_top_terms(self, n=10):
        """
        Mendapatkan n term teratas untuk setiap dokumen.
        
        Parameters:
        -----------
        n : int, default=10
            Jumlah term teratas yang akan diambil
            
        Returns:
        --------
        dict
            Dictionary berisi n term teratas untuk setiap dokumen
        """
        if self.tfidf_matrix is None:
            raise ValueError("Fit the model first before getting top terms")
        
        start_time = time.time()
        
        top_terms = {}
        for doc_idx, doc_name in enumerate(self.document_names):
            # Mendapatkan nilai TF-IDF untuk dokumen ini
            tfidf_scores = self.tfidf_matrix[doc_idx].toarray().flatten()
            
            # Mendapatkan indeks term dengan nilai TF-IDF tertinggi
            top_indices = tfidf_scores.argsort()[-n:][::-1]
            
            # Mendapatkan term dan nilai TF-IDF
            top_terms[doc_name] = {
                'terms': self.feature_names[top_indices],
                'scores': tfidf_scores[top_indices]
            }
        
        self.top_terms = top_terms
        self.execution_time['top_terms'] = time.time() - start_time
        
        return top_terms
    
    def calculate_document_similarity(self):
        """
        Menghitung kemiripan antar dokumen menggunakan cosine similarity.
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame berisi nilai kemiripan antar dokumen
        """
        start_time = time.time()
        
        # Menghitung cosine similarity
        cosine_sim = cosine_similarity(self.tfidf_matrix)
        
        # Membuat DataFrame untuk hasil
        similarity_df = pd.DataFrame(
            cosine_sim, 
            index=self.document_names, 
            columns=self.document_names
        )
        
        self.execution_time['document_similarity'] = time.time() - start_time
        logger.info(f"Document similarity calculation completed in {self.execution_time['document_similarity']:.2f} seconds")
        
        return similarity_df
    
    def cluster_documents(self, n_clusters=5):
        """
        Mengelompokkan dokumen menggunakan K-means clustering.
        
        Parameters:
        -----------
        n_clusters : int, default=5
            Jumlah cluster
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame berisi informasi cluster untuk setiap dokumen
        """
        start_time = time.time()
        
        # Menjalankan K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(self.tfidf_matrix)
        
        # Membuat DataFrame untuk hasil
        cluster_df = pd.DataFrame({
            'Document': self.document_names,
            'Cluster': clusters
        })
        
        self.document_clusters = cluster_df
        self.execution_time['clustering'] = time.time() - start_time
        logger.info(f"Document clustering completed in {self.execution_time['clustering']:.2f} seconds")
        
        return cluster_df
    
    def visualize_clusters(self, method='tsne'):
   
        """
        Memvisualisasikan cluster dokumen.
        Parameters:
        -----------
        method : str, default='tsne'
            Metode dimensionality reduction ('tsne', 'pca', 'svd')
        Returns:
        --------
        matplotlib.figure.Figure
            Figure hasil visualisasi
        """
        start_time = time.time()
        if self.document_clusters is None:
            raise ValueError("Run cluster_documents() first before visualizing clusters")
        
        # Mengurangi dimensi untuk visualisasi
        if method == 'tsne':
            # Gunakan init='random' dan perplexity < n_samples (10)
            reducer = TSNE(
                n_components=2, 
                random_state=42, 
                perplexity=5,  # Perbaikan: perplexity < 10
                init='random'    # Perbaikan: Hindari PCA untuk matriks sparse
            )
        elif method == 'pca':
            reducer = PCA(n_components=2, random_state=42)
        elif method == 'svd':
            reducer = TruncatedSVD(n_components=2, random_state=42)
        else:
            raise ValueError("Method must be one of 'tsne', 'pca', or 'svd'")
        
        reduced_features = reducer.fit_transform(self.tfidf_matrix)
    # ... (sisa kode tetap sama)
        
        # Membuat DataFrame untuk plotting
        cluster_viz_df = pd.DataFrame({
            'Document': self.document_names,
            'Cluster': self.document_clusters['Cluster'],
            'x': reduced_features[:, 0],
            'y': reduced_features[:, 1]
        })
        
        # Plotting
        plt.figure(figsize=(12, 8))
        
        # Pallete warna
        palette = sns.color_palette("husl", len(cluster_viz_df['Cluster'].unique()))
        
        # Membuat scatter plot
        ax = sns.scatterplot(
            x='x', y='y',
            hue='Cluster',
            palette=palette,
            data=cluster_viz_df,
            legend='full',
            alpha=0.7,
            s=100
        )
        
        # Menambahkan label untuk setiap titik
        for i, row in cluster_viz_df.iterrows():
            plt.annotate(
                row['Document'],
                (row['x'], row['y']),
                fontsize=9,
                alpha=0.75,
                xytext=(5, 5),
                textcoords='offset points'
            )
        
        plt.title(f'Document Clusters ({method.upper()})')
        plt.tight_layout()
        
        self.execution_time['visualize_clusters'] = time.time() - start_time
        logger.info(f"Cluster visualization completed in {self.execution_time['visualize_clusters']:.2f} seconds")
        
        return plt.gcf()
    
    def create_term_frequency_plot(self, top_n=20):
        """
        Membuat visualisasi frekuensi term.
        
        Parameters:
        -----------
        top_n : int, default=20
            Jumlah term teratas yang akan divisualisasikan
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure hasil visualisasi
        """
        start_time = time.time()
        
        # Menggabungkan semua dokumen yang telah diproses
        all_text = ' '.join(self.processed_documents)
        
        # Menghitung frekuensi kata
        words = all_text.split()
        word_counts = Counter(words)
        
        # Mendapatkan top_n kata dengan frekuensi tertinggi
        top_words = dict(word_counts.most_common(top_n))
        
        # Membuat plot
        plt.figure(figsize=(14, 10))
        
        # Plot bar untuk frekuensi kata
        plt.subplot(2, 1, 1)
        sns.barplot(x=list(top_words.keys()), y=list(top_words.values()), palette='viridis')
        plt.title(f'Top {top_n} Term Frequencies')
        plt.xticks(rotation=45, ha='right')
        plt.xlabel('Term')
        plt.ylabel('Frequency')
        
        # Plot word cloud
        plt.subplot(2, 1, 2)
        wordcloud = WordCloud(
            width=800, height=400,
            background_color='white',
            colormap='viridis',
            max_words=100,
            contour_width=3,
            contour_color='steelblue'
        ).generate_from_frequencies(word_counts)
        
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Term Frequency Word Cloud')
        
        plt.tight_layout()
        
        self.execution_time['term_frequency_plot'] = time.time() - start_time
        
        return plt.gcf()
    
    def visualize_document_similarity(self):
        """
        Memvisualisasikan kemiripan antar dokumen sebagai heatmap dan graph.
        
        Returns:
        --------
        matplotlib.figure.Figure
            Figure hasil visualisasi
        """
        start_time = time.time()
        
        # Menghitung similarity matrix
        similarity_matrix = cosine_similarity(self.tfidf_matrix)
        similarity_df = pd.DataFrame(
            similarity_matrix,
            index=self.document_names,
            columns=self.document_names
        )
        
        # Membuat visualisasi
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Heatmap
        sns.heatmap(
            similarity_df,
            annot=True,
            cmap='YlGnBu',
            ax=ax1,
            fmt='.2f',
            linewidths=0.5,
            cbar_kws={'label': 'Cosine Similarity'}
        )
        ax1.set_title('Document Similarity Heatmap')
        
        # Network graph
        G = nx.Graph()
        
        # Menambahkan node
        for doc in self.document_names:
            G.add_node(doc)
        
        # Menambahkan edge dengan bobot berdasarkan similarity
        for i, doc1 in enumerate(self.document_names):
            for j, doc2 in enumerate(self.document_names):
                if i < j:  # Untuk menghindari duplikasi edge
                    sim = similarity_matrix[i, j]
                    if sim > 0.1:  # Hanya menampilkan edge dengan similarity > 0.1
                        G.add_edge(doc1, doc2, weight=sim)
        
        # Posisi node menggunakan spring layout
        pos = nx.spring_layout(G, seed=42)
        
        # Menggambar node
        nx.draw_networkx_nodes(
            G, pos,
            node_size=500,
            node_color='lightblue',
            alpha=0.8,
            ax=ax2
        )
        
        # Menggambar edge dengan warna berdasarkan bobot
        edges = nx.draw_networkx_edges(
            G, pos,
            width=[G[u][v]['weight'] * 5 for u, v in G.edges()],
            alpha=0.5,
            edge_color=[G[u][v]['weight'] for u, v in G.edges()],
            edge_cmap=plt.cm.YlGnBu,
            ax=ax2
        )
        
        # Menambahkan label
        nx.draw_networkx_labels(
            G, pos,
            font_size=10,
            font_family='sans-serif',
            ax=ax2
        )
        
        ax2.set_title('Document Similarity Network')
        ax2.axis('off')
        
        plt.tight_layout()
        
        self.execution_time['document_similarity_viz'] = time.time() - start_time
        
        return plt.gcf()
    
    def visualize_tfidf_heatmap(self, top_n=20):
        """
        Membuat heatmap nilai TF-IDF untuk term teratas.
        
        Parameters:
        -----------
        top_n : int, default=20
            Jumlah term teratas yang akan divisualisasikan
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure hasil visualisasi
        """
        start_time = time.time()
        
        # Mendapatkan term dengan rata-rata nilai TF-IDF tertinggi
        tfidf_means = np.mean(self.tfidf_matrix.toarray(), axis=0)
        top_indices = tfidf_means.argsort()[-top_n:][::-1]
        top_terms = self.feature_names[top_indices]
        
        # Membuat DataFrame untuk heatmap
        tfidf_df = pd.DataFrame(
            self.tfidf_matrix.toarray()[:, top_indices],
            index=self.document_names,
            columns=top_terms
        )
        
        # Membuat heatmap
        plt.figure(figsize=(14, 10))
        sns.heatmap(
            tfidf_df,
            annot=True,
            cmap='YlOrRd',
            fmt='.2f',
            linewidths=0.5,
            cbar_kws={'label': 'TF-IDF Score'}
        )
        plt.title(f'TF-IDF Heatmap for Top {top_n} Terms')
        plt.tight_layout()
        
        self.execution_time['tfidf_heatmap'] = time.time() - start_time
        
        return plt.gcf()
    
    def visualize_tfidf_distribution(self):
        """
        Memvisualisasikan distribusi nilai TF-IDF.
        
        Returns:
        --------
        matplotlib.figure.Figure
            Figure hasil visualisasi
        """
        start_time = time.time()
        
        # Mendapatkan nilai TF-IDF
        tfidf_values = self.tfidf_matrix.toarray().flatten()
        tfidf_values = tfidf_values[tfidf_values > 0]  # Hanya nilai positif
        
        # Membuat plot
        plt.figure(figsize=(14, 10))
        
        # Histogram
        plt.subplot(2, 1, 1)
        sns.histplot(tfidf_values, kde=True, bins=50, color='steelblue')
        plt.title('TF-IDF Score Distribution')
        plt.xlabel('TF-IDF Score')
        plt.ylabel('Frequency')
        
        # Box plot untuk setiap dokumen
        plt.subplot(2, 1, 2)
        tfidf_by_doc = [doc.data for doc in self.tfidf_matrix]
        plt.boxplot(tfidf_by_doc, labels=self.document_names)
        plt.title('TF-IDF Score Distribution by Document')
        plt.xlabel('Document')
        plt.ylabel('TF-IDF Score')
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        self.execution_time['tfidf_distribution'] = time.time() - start_time
        
        return plt.gcf()
    
    def visualize_term_importance(self, doc_idx=0, top_n=20):
        """
        Memvisualisasikan term terpenting dalam dokumen tertentu.
        
        Parameters:
        -----------
        doc_idx : int, default=0
            Indeks dokumen
        top_n : int, default=20
            Jumlah term teratas yang akan divisualisasikan
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure hasil visualisasi
        """
        start_time = time.time()
        
        # Mendapatkan nilai TF-IDF untuk dokumen tertentu
        tfidf_scores = self.tfidf_matrix[doc_idx].toarray().flatten()
        
        # Mendapatkan term dengan nilai TF-IDF tertinggi
        top_indices = tfidf_scores.argsort()[-top_n:][::-1]
        top_terms = self.feature_names[top_indices]
        top_scores = tfidf_scores[top_indices]
        
        # Membuat DataFrame untuk plotting
        term_importance_df = pd.DataFrame({
            'Term': top_terms,
            'TF-IDF Score': top_scores
        })
        
        # Membuat plot
        plt.figure(figsize=(14, 10))
        
        # Bar plot
        plt.subplot(2, 1, 1)
        sns.barplot(
            x='TF-IDF Score',
            y='Term',
            data=term_importance_df,
            palette='viridis'
        )
        plt.title(f'Top {top_n} Important Terms in {self.document_names[doc_idx]}')
        plt.xlabel('TF-IDF Score')
        plt.ylabel('Term')
        
        # Word cloud
        plt.subplot(2, 1, 2)
        term_dict = dict(zip(term_importance_df['Term'], term_importance_df['TF-IDF Score']))
        wordcloud = WordCloud(
            width=800, height=400,
            background_color='white',
            colormap='viridis',
            max_words=100,
            contour_width=3,
            contour_color='steelblue'
        ).generate_from_frequencies(term_dict)
        
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Word Cloud for {self.document_names[doc_idx]}')
        
        plt.tight_layout()
        
        self.execution_time['term_importance'] = time.time() - start_time
        
        return plt.gcf()
    
    def create_performance_report(self):
        """
        Membuat laporan performa dari berbagai operasi.
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame berisi waktu eksekusi berbagai operasi
        """
        operations = list(self.execution_time.keys())
        times = list(self.execution_time.values())
        
        performance_df = pd.DataFrame({
            'Operation': operations,
            'Execution Time (s)': times
        })
        
        performance_df = performance_df.sort_values('Execution Time (s)', ascending=False)
        
        return performance_df
    
    def visualize_performance(self):
        """
        Memvisualisasikan performa berbagai operasi.
        
        Returns:
        --------
        matplotlib.figure.Figure
            Figure hasil visualisasi
        """
        performance_df = self.create_performance_report()
        
        plt.figure(figsize=(12, 8))
        sns.barplot(
            x='Execution Time (s)',
            y='Operation',
            data=performance_df,
            palette='viridis'
        )
        plt.title('Execution Time for Various Operations')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Operation')
        plt.tight_layout()
        
        return plt.gcf()
    
    def save_model(self, filepath):
        """
        Menyimpan model ke file.
        
        Parameters:
        -----------
        filepath : str
            Path ke file untuk menyimpan model
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        logger.info(f"Model saved to {filepath}")
    
    @staticmethod
    def load_model(filepath):
        """
        Memuat model dari file.
        
        Parameters:
        -----------
        filepath : str
            Path ke file model
            
        Returns:
        --------
        TFIDFAnalyzer
            Model yang dimuat
        """
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Model loaded from {filepath}")
        return model
    
    def interactive_tfidf_explorer(self):
        """
        Membuat visualisasi interaktif untuk eksplorasi TF-IDF.
        
        Returns:
        --------
        plotly.graph_objects.Figure
            Figure interaktif
        """
        # Mendapatkan matriks TF-IDF sebagai array
        tfidf_array = self.tfidf_matrix.toarray()
        
        # Mendapatkan 50 term dengan variance tertinggi
        variances = np.var(tfidf_array, axis=0)
        top_var_indices = variances.argsort()[-50:][::-1]
        top_var_terms = self.feature_names[top_var_indices]
        
        # Membuat DataFrame untuk visualisasi
        tfidf_df = pd.DataFrame(
            tfidf_array[:, top_var_indices],
            index=self.document_names,
            columns=top_var_terms
        )
        
        # Membuat figure interaktif
        fig = px.imshow(
            tfidf_df,
            labels=dict(x="Term", y="Document", color="TF-IDF Score"),
            x=top_var_terms,
            y=self.document_names,
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(
            title="Interactive TF-IDF Heatmap",
            xaxis_title="Term",
            yaxis_title="Document",
            height=600,
            width=1000
        )
        
        return fig

# Contoh penggunaan
def demo_tfidf_analyzer():
    """
    Mendemonstrasikan penggunaan TFIDFAnalyzer dengan contoh dokumen.
    
    Returns:
    --------
    TFIDFAnalyzer
        Instance TFIDFAnalyzer yang sudah dilatih
    """
    # Contoh dokumen
    documents = [
        "TF-IDF stands for Term Frequency-Inverse Document Frequency. It's a numerical statistic that is intended to reflect how important a word is to a document in a collection or corpus.",
        "The TF-IDF weight is a weight often used in information retrieval and text mining. This weight is a statistical measure used to evaluate how important a word is to a document in a collection or corpus.",
        "Term frequency is simply the number of times a term appears in a document. Inverse document frequency measures how common or rare a word is across all documents.",
        "Text mining is the process of deriving meaningful information from natural language text. It uses techniques from natural language processing, information retrieval, and machine learning.",
        "Information retrieval is the science of searching for information in documents, searching for documents themselves, and also searching for metadata that describes data.",
        "Machine learning algorithms can learn from and make predictions on data. They use statistics to find patterns in massive amounts of data.",
        "Natural language processing is a field of artificial intelligence that gives computers the ability to understand text and spoken words in the same way humans can.",
        "Python is a popular programming language for data science and machine learning. It has many libraries for text processing, such as NLTK and scikit-learn.",
        "Data visualization is the graphical representation of information and data. Visual elements like charts, graphs, and maps make it easier to understand patterns and trends in data.",
        "Clustering algorithms group similar documents together based on their content. K-means is a popular clustering algorithm used in text mining."
    ]
    
    # Nama dokumen
    document_names = [
        "Definition TF-IDF",
        "Usage of TF-IDF",
        "Term Frequency",
        "Text Mining",
        "Information Retrieval",
        "Machine Learning",
        "NLP",
        "Python",
        "Data Visualization",
        "Clustering"
    ]
    
    # Inisialisasi dan latih analyzer
    analyzer = TFIDFAnalyzer(
        language='english',
        custom_stopwords=['the', 'is', 'a', 'an', 'and', 'in', 'of', 'to', 'for'],
        min_df=1,
        max_df=0.9,
        ngram_range=(1, 2),
        use_lemmatization=True
    )
    
    analyzer.fit(documents, document_names)
    
    # Mendapatkan term teratas untuk setiap dokumen
    top_terms = analyzer.get_top_terms(n=5)
    
    # Menghitung kemiripan antar dokumen
    similarity_df = analyzer.calculate_document_similarity()
    
    # Mengelompokkan dokumen
    cluster_df = analyzer.cluster_documents(n_clusters=3)
    
    # Visualisasi
    cluster_viz = analyzer.visualize_clusters(method='tsne')
    term_freq_viz = analyzer.create_term_frequency_plot(top_n=15)
    sim_viz = analyzer.visualize_document_similarity()
    heatmap_viz = analyzer.visualize_tfidf_heatmap(top_n=15)
    dist_viz = analyzer.visualize_tfidf_distribution()
    term_imp_viz = analyzer.visualize_term_importance(doc_idx=0, top_n=15)
    
    # Membuat laporan performa
    perf_df = analyzer.create_performance_report()
    perf_viz = analyzer.visualize_performance()
    
    # Interactive explorer
    interactive_viz = analyzer.interactive_tfidf_explorer()
    
    return analyzer

if __name__ == "__main__":
    analyzer = demo_tfidf_analyzer()

# Kelas untuk analisis TF-IDF lanjutan dengan fitur tambahan
class AdvancedTFIDFAnalyzer(TFIDFAnalyzer):
    """
    Kelas untuk analisis TF-IDF lanjutan dengan fitur tambahan.
    """
    
    def __init__(self, **kwargs):
        """
        Inisialisasi AdvancedTFIDFAnalyzer.
        
        Parameters:
        -----------
        **kwargs : dict
            Parameter yang diteruskan ke TFIDFAnalyzer
        """
        super().__init__(**kwargs)
        self.topic_model = None
        self.topic_terms = None
        self.topic_assignments = None
    
    def extract_topics(self, n_topics=5, n_top_words=10, method='nmf'):
        """
        Mengekstrak topik dari dokumen menggunakan metode topic modeling.
        
        Parameters:
        -----------
        n_topics : int, default=5
            Jumlah topik yang akan diekstrak
        n_top_words : int, default=10
            Jumlah kata teratas untuk setiap topik
        method : str, default='nmf'
            Metode topic modeling ('nmf', 'lda')
            
        Returns:
        --------
        dict
            Dictionary berisi term teratas untuk setiap topik
        """
        start_time = time.time()
        
        if method == 'nmf':
            from sklearn.decomposition import NMF
            model = NMF(n_components=n_topics, random_state=42)
        elif method == 'lda':
            from sklearn.decomposition import LatentDirichletAllocation
            model = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        else:
            raise ValueError("Method must be one of 'nmf' or 'lda'")
        
        # Melatih model
        W = model.fit_transform(self.tfidf_matrix)
        H = model.components_
        
        # Menyimpan model
        self.topic_model = model
        
        # Mendapatkan term teratas untuk setiap topik
        topic_terms = {}
        for topic_idx, topic in enumerate(H):
            top_term_indices = topic.argsort()[:-n_top_words-1:-1]
            top_terms = self.feature_names[top_term_indices]
            topic_terms[f"Topic {topic_idx+1}"] = top_terms
        
        self.topic_terms = topic_terms
        
        # Menetapkan topik untuk setiap dokumen
        doc_topic_assignments = []
        for i, doc_name in enumerate(self.document_names):
            topic_idx = W[i].argmax()
            doc_topic_assignments.append({
                'Document': doc_name,
                'Topic': f"Topic {topic_idx+1}",
                'Score': W[i, topic_idx]
            })
        
        self.topic_assignments = pd.DataFrame(doc_topic_assignments)
        
        self.execution_time['topic_extraction'] = time.time() - start_time
        logger.info(f"Topic extraction completed in {self.execution_time['topic_extraction']:.2f} seconds")
        
        return topic_terms
    
    def visualize_topics(self):
        """
        Memvisualisasikan topik yang diekstrak.
        
        Returns:
        --------
        matplotlib.figure.Figure
            Figure hasil visualisasi
        """
        if self.topic_terms is None:
            raise ValueError("Run extract_topics() first before visualizing topics")
        
        start_time = time.time()
        
        # Jumlah topik dan term per topik
        n_topics = len(self.topic_terms)
        n_terms = len(next(iter(self.topic_terms.values())))
        
        # Membuat colormap
        colors = plt.cm.tab10(np.linspace(0, 1, n_topics))
        
        # Membuat plot
        fig, axes = plt.subplots(n_topics, 1, figsize=(15, 5*n_topics))
        
        for i, (topic_name, terms) in enumerate(self.topic_terms.items()):
            ax = axes[i] if n_topics > 1 else axes
            
            y_pos = np.arange(len(terms))
            ax.barh(y_pos, np.linspace(1, 0.5, len(terms)), align='center', color=colors[i], alpha=0.7)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(terms)
            ax.invert_yaxis()
            ax.set_xlabel('Importance')
            ax.set_title(f'{topic_name}')
        
        plt.tight_layout()
        
        self.execution_time['visualize_topics'] = time.time() - start_time
        
        return plt.gcf()
    
    def visualize_document_topic_distribution(self):
        """
        Memvisualisasikan distribusi topik untuk setiap dokumen.
        
        Returns:
        --------
        matplotlib.figure.Figure
            Figure hasil visualisasi
        """
        if self.topic_assignments is None:
            raise ValueError("Run extract_topics() first before visualizing topic distribution")
        
        start_time = time.time()
        
        # Menghitung jumlah dokumen per topik
        topic_counts = self.topic_assignments['Topic'].value_counts()
        
        # Membuat plot
        plt.figure(figsize=(12, 10))
        
        # Pie chart
        plt.subplot(2, 1, 1)
        plt.pie(
            topic_counts,
            labels=topic_counts.index,
            autopct='%1.1f%%',
            shadow=True,
            startangle=90,
            colors=plt.cm.tab10(np.linspace(0, 1, len(topic_counts)))
        )
        plt.axis('equal')
        plt.title('Document Distribution by Topic')
        
        # Bar chart
        plt.subplot(2, 1, 2)
        sns.barplot(
            x='Topic',
            y='Score',
            data=self.topic_assignments,
            hue='Document',
            palette='viridis'
        )
        plt.title('Document-Topic Assignment Scores')
        plt.xticks(rotation=45, ha='right')
        plt.xlabel('Topic')
        plt.ylabel('Assignment Score')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        self.execution_time['visualize_doc_topic'] = time.time() - start_time
        
        return plt.gcf()
    
    def calculate_topic_coherence(self):
        """
        Menghitung koherensi topik.
        
        Returns:
        --------
        float
            Nilai koherensi
        """
        if self.topic_terms is None:
            raise ValueError("Run extract_topics() first before calculating coherence")
        
        start_time = time.time()
        
        # Menggunakan ukuran koherensi sederhana berdasarkan co-occurrence
        coherence_scores = []
        
        # Untuk setiap pasangan term dalam topik yang sama
        for topic_name, terms in self.topic_terms.items():
            topic_coherence = 0
            term_pairs = 0
            
            for i, term1 in enumerate(terms):
                for j, term2 in enumerate(terms):
                    if i < j:  # Untuk menghindari duplikasi
                        # Menghitung dokumen yang memiliki term1
                        term1_docs = set()
                        term1_idx = np.where(self.feature_names == term1)[0]
                        if len(term1_idx) > 0:
                            term1_idx = term1_idx[0]
                            term1_docs = set(np.where(self.tfidf_matrix.toarray()[:, term1_idx] > 0)[0])
                        
                        # Menghitung dokumen yang memiliki term2
                        term2_docs = set()
                        term2_idx = np.where(self.feature_names == term2)[0]
                        if len(term2_idx) > 0:
                            term2_idx = term2_idx[0]
                            term2_docs = set(np.where(self.tfidf_matrix.toarray()[:, term2_idx] > 0)[0])
                        
                        # Dokumen yang memiliki kedua term
                        co_docs = term1_docs.intersection(term2_docs)
                        
                        # Menghitung koherensi menggunakan PMI
                        if len(term1_docs) > 0 and len(term2_docs) > 0:
                            prob_term1 = len(term1_docs) / len(self.documents)
                            prob_term2 = len(term2_docs) / len(self.documents)
                            prob_co = (len(co_docs) + 1) / len(self.documents)  # Laplace smoothing
                            
                            pmi = np.log(prob_co / (prob_term1 * prob_term2))
                            topic_coherence += pmi
                            term_pairs += 1
            
            if term_pairs > 0:
                coherence_scores.append(topic_coherence / term_pairs)
        
        coherence = np.mean(coherence_scores)
        
        self.execution_time['topic_coherence'] = time.time() - start_time
        logger.info(f"Topic coherence calculation completed in {self.execution_time['topic_coherence']:.2f} seconds")

        return coherence
    
    def create_semantic_network(self):
        """
        Membuat jaringan semantik berdasarkan co-occurrence term.
        
        Returns:
        --------
        matplotlib.figure.Figure
            Figure hasil visualisasi
        """
        start_time = time.time()
        
        # Menggunakan top 50 term dengan nilai TF-IDF tertinggi
        tfidf_sums = np.sum(self.tfidf_matrix.toarray(), axis=0)
        top_indices = tfidf_sums.argsort()[-50:][::-1]
        top_terms = self.feature_names[top_indices]
        
        # Membuat graph
        G = nx.Graph()
        
        # Menambahkan node
        for term in top_terms:
            G.add_node(term)
        
        # Menambahkan edge berdasarkan co-occurrence
        for i, term1 in enumerate(top_terms):
            term1_idx = np.where(self.feature_names == term1)[0][0]
            term1_docs = set(np.where(self.tfidf_matrix.toarray()[:, term1_idx] > 0)[0])
            
            for j, term2 in enumerate(top_terms):
                if i < j:  # Untuk menghindari duplikasi
                    term2_idx = np.where(self.feature_names == term2)[0][0]
                    term2_docs = set(np.where(self.tfidf_matrix.toarray()[:, term2_idx] > 0)[0])
                    
                    # Dokumen yang memiliki kedua term
                    co_docs = term1_docs.intersection(term2_docs)
                    
                    if len(co_docs) > 0:
                        # Menghitung bobot edge berdasarkan jumlah co-occurrence
                        weight = len(co_docs) / len(self.documents)
                        G.add_edge(term1, term2, weight=weight)
        
        # Membuat plot
        plt.figure(figsize=(15, 12))
        
        # Posisi node menggunakan spring layout
        pos = nx.spring_layout(G, seed=42, k=0.3)
        
        # Node sizes berdasarkan nilai TF-IDF
        node_sizes = []
        for term in G.nodes():
            term_idx = np.where(self.feature_names == term)[0][0]
            tfidf_sum = np.sum(self.tfidf_matrix.toarray()[:, term_idx])
            node_sizes.append(300 * tfidf_sum + 100)  # Menskalakan ukuran node
        
        # Edge widths berdasarkan bobot
        edge_widths = [G[u][v]['weight'] * 5 for u, v in G.edges()]
        
        # Menggambar node
        nx.draw_networkx_nodes(
            G, pos,
            node_size=node_sizes,
            node_color=range(len(G.nodes())),
            cmap=plt.cm.viridis,
            alpha=0.8
        )
        
        # Menggambar edge
        nx.draw_networkx_edges(
            G, pos,
            width=edge_widths,
            alpha=0.5,
            edge_color='gray'
        )
        
        # Menambahkan label
        nx.draw_networkx_labels(
            G, pos,
            font_size=10,
            font_family='sans-serif'
        )
        
        plt.title('Semantic Network of Terms')
        plt.axis('off')
        plt.tight_layout()
        
        self.execution_time['semantic_network'] = time.time() - start_time
        
        return plt.gcf()
    
    def analyze_sentiment(self):
        """
        Melakukan analisis sentimen sederhana pada dokumen.
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame berisi skor sentimen untuk setiap dokumen
        """
        start_time = time.time()
        
        # Daftar kata positif dan negatif sederhana
        positive_words = {
            'good', 'great', 'excellent', 'positive', 'best', 'better', 'important',
            'useful', 'helpful', 'effective', 'valuable', 'interesting', 'powerful',
            'innovative', 'advanced', 'efficient', 'reliable', 'robust', 'significant'
        }
        
        negative_words = {
            'bad', 'worse', 'worst', 'negative', 'difficult', 'hard', 'complex',
            'problematic', 'challenging', 'inefficient', 'unreliable', 'poor',
            'limited', 'weak', 'complicated', 'expensive', 'confusing'
        }
        
        # Menghitung skor sentimen untuk setiap dokumen
        sentiment_scores = []
        
        for doc_idx, doc_name in enumerate(self.document_names):
            # Mendapatkan nilai TF-IDF untuk dokumen ini
            doc_vector = self.tfidf_matrix[doc_idx].toarray().flatten()
            
            positive_score = 0
            negative_score = 0
            
            for term_idx, term in enumerate(self.feature_names):
                term_score = doc_vector[term_idx]
                
                if term in positive_words:
                    positive_score += term_score
                elif term in negative_words:
                    negative_score += term_score
            
            # Menghitung skor sentimen total
            total_score = positive_score - negative_score
            
            sentiment_scores.append({
                'Document': doc_name,
                'Positive Score': positive_score,
                'Negative Score': negative_score,
                'Total Score': total_score,
                'Sentiment': 'Positive' if total_score > 0 else 'Negative' if total_score < 0 else 'Neutral'
            })
        
        sentiment_df = pd.DataFrame(sentiment_scores)
        
        self.execution_time['sentiment_analysis'] = time.time() - start_time
        logger.info(f"Sentiment analysis completed in {self.execution_time['sentiment_analysis']:.2f} seconds")
        
        return sentiment_df
    
    def visualize_sentiment(self, sentiment_df):
        """
        Memvisualisasikan hasil analisis sentimen.
        
        Parameters:
        -----------
        sentiment_df : pandas.DataFrame
            DataFrame hasil analisis sentimen
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure hasil visualisasi
        """
        start_time = time.time()
        
        # Membuat plot
        plt.figure(figsize=(15, 10))
        
        # Bar chart untuk skor sentimen
        plt.subplot(2, 1, 1)
        
        # Menyusun data untuk plotting
        pos_scores = sentiment_df['Positive Score'].values
        neg_scores = -sentiment_df['Negative Score'].values
        
        x = np.arange(len(sentiment_df))
        width = 0.35
        
        plt.bar(x - width/2, pos_scores, width, label='Positive', color='green', alpha=0.7)
        plt.bar(x + width/2, neg_scores, width, label='Negative', color='red', alpha=0.7)
        
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.xticks(x, sentiment_df['Document'], rotation=45, ha='right')
        plt.xlabel('Document')
        plt.ylabel('Sentiment Score')
        plt.title('Positive and Negative Sentiment Scores by Document')
        plt.legend()
        
        # Pie chart untuk distribusi sentimen
        plt.subplot(2, 1, 2)
        sentiment_counts = sentiment_df['Sentiment'].value_counts()
        colors = {'Positive': 'green', 'Negative': 'red', 'Neutral': 'gray'}
        sentiment_colors = [colors[s] for s in sentiment_counts.index]
        
        plt.pie(
            sentiment_counts,
            labels=sentiment_counts.index,
            autopct='%1.1f%%',
            startangle=90,
            colors=sentiment_colors,
            explode=[0.1 if s == sentiment_counts.index[0] else 0 for s in sentiment_counts.index]
        )
        plt.axis('equal')
        plt.title('Document Sentiment Distribution')
        
        plt.tight_layout()
        
        self.execution_time['visualize_sentiment'] = time.time() - start_time
        
        return plt.gcf()
    
    def create_comprehensive_report(self):
        """
        Membuat laporan komprehensif dari semua analisis.
        
        Returns:
        --------
        dict
            Dictionary berisi berbagai hasil analisis
        """
        start_time = time.time()
        
        # Daftar hasil
        results = {}
        
        # Informasi dasar
        results['basic_info'] = {
            'num_documents': len(self.documents),
            'num_terms': len(self.feature_names),
            'vocabulary_size': self.tfidf_matrix.shape[1],
            'sparsity': 1.0 - (np.count_nonzero(self.tfidf_matrix.toarray()) / 
                             (self.tfidf_matrix.shape[0] * self.tfidf_matrix.shape[1]))
        }
        
        # Top terms
        results['top_terms'] = self.get_top_terms(n=10)
        
        # Document similarity
        results['document_similarity'] = self.calculate_document_similarity()
        
        # Clustering
        results['document_clusters'] = self.cluster_documents(n_clusters=min(3, len(self.documents)))
        
        # Performance
        results['performance'] = self.create_performance_report()
        
        # Jika telah dilakukan ekstraksi topik
        if hasattr(self, 'topic_terms') and self.topic_terms is not None:
            results['topics'] = self.topic_terms
            results['topic_assignments'] = self.topic_assignments
        
        # Jika telah dilakukan analisis sentimen
        try:
            sentiment_df = self.analyze_sentiment()
            results['sentiment'] = sentiment_df
        except:
            pass
        
        self.execution_time['comprehensive_report'] = time.time() - start_time
        logger.info(f"Comprehensive report created in {self.execution_time['comprehensive_report']:.2f} seconds")
        
        return results

def demo_advanced_analyzer():
    """
    Mendemonstrasikan penggunaan AdvancedTFIDFAnalyzer.
    
    Returns:
    --------
    AdvancedTFIDFAnalyzer
        Instance AdvancedTFIDFAnalyzer yang sudah dilatih
    """
    # Contoh dokumen yang sama dengan demo sebelumnya
    documents = [
        "TF-IDF stands for Term Frequency-Inverse Document Frequency. It's a numerical statistic that is intended to reflect how important a word is to a document in a collection or corpus.",
        "The TF-IDF weight is a weight often used in information retrieval and text mining. This weight is a statistical measure used to evaluate how important a word is to a document in a collection or corpus.",
        "Term frequency is simply the number of times a term appears in a document. Inverse document frequency measures how common or rare a word is across all documents.",
        "Text mining is the process of deriving meaningful information from natural language text. It uses techniques from natural language processing, information retrieval, and machine learning.",
        "Information retrieval is the science of searching for information in documents, searching for documents themselves, and also searching for metadata that describes data.",
        "Machine learning algorithms can learn from and make predictions on data. They use statistics to find patterns in massive amounts of data.",
        "Natural language processing is a field of artificial intelligence that gives computers the ability to understand text and spoken words in the same way humans can.",
        "Python is a popular programming language for data science and machine learning. It has many libraries for text processing, such as NLTK and scikit-learn.",
        "Data visualization is the graphical representation of information and data. Visual elements like charts, graphs, and maps make it easier to understand patterns and trends in data.",
        "Clustering algorithms group similar documents together based on their content. K-means is a popular clustering algorithm used in text mining."
    ]
    
    # Nama dokumen
    document_names = [
        "Definition TF-IDF",
        "Usage of TF-IDF",
        "Term Frequency",
        "Text Mining",
        "Information Retrieval",
        "Machine Learning",
        "NLP",
        "Python",
        "Data Visualization",
        "Clustering"
    ]
    
    # Inisialisasi dan latih analyzer
    advanced_analyzer = AdvancedTFIDFAnalyzer(
        language='english',
        custom_stopwords=['the', 'is', 'a', 'an', 'and', 'in', 'of', 'to', 'for'],
        min_df=1,
        max_df=0.9,
        ngram_range=(1, 2),
        use_lemmatization=True
    )
    
    advanced_analyzer.fit(documents, document_names)
    
    # Ekstraksi topik
    topics = advanced_analyzer.extract_topics(n_topics=3, n_top_words=8, method='nmf')
    
    # Visualisasi topik
    topic_viz = advanced_analyzer.visualize_topics()
    topic_dist_viz = advanced_analyzer.visualize_document_topic_distribution()
    
    # Koherensi topik
    coherence = advanced_analyzer.calculate_topic_coherence()
    print(f"Topic coherence: {coherence:.4f}")
    
    # Jaringan semantik
    semantic_network = advanced_analyzer.create_semantic_network()
    
    # Analisis sentimen
    sentiment_df = advanced_analyzer.analyze_sentiment()
    sentiment_viz = advanced_analyzer.visualize_sentiment(sentiment_df)
    
    # Laporan komprehensif
    report = advanced_analyzer.create_comprehensive_report()
    
    return advanced_analyzer

# Kelas untuk pemrosesan TF-IDF real-time
class StreamingTFIDFProcessor:
    """
    Kelas untuk pemrosesan TF-IDF real-time pada aliran dokumen.
    """
    
    def __init__(self, base_analyzer):
        """
        Inisialisasi StreamingTFIDFProcessor.
        
        Parameters:
        -----------
        base_analyzer : TFIDFAnalyzer
            Analyzer dasar yang digunakan sebagai model awal
        """
        self.base_analyzer = base_analyzer
        self.documents = list(base_analyzer.documents)
        self.document_names = list(base_analyzer.document_names)
        self.processed_documents = list(base_analyzer.processed_documents)
        self.tfidf_matrix = base_analyzer.tfidf_matrix.copy()
        self.feature_names = base_analyzer.feature_names
        self.stream_history = []
    
    def process_new_document(self, new_document, document_name=None):
        """
        Memproses dokumen baru dan memperbarui model TF-IDF.
        
        Parameters:
        -----------
        new_document : str
            Dokumen baru
        document_name : str, default=None
            Nama dokumen baru
            
        Returns:
        --------
        dict
            Dictionary berisi informasi tentang dokumen baru
        """
        # Preprocess dokumen baru
        processed_document = self.base_analyzer.preprocess_text(new_document)
        
        # Menyimpan dokumen
        self.documents.append(new_document)
        self.processed_documents.append(processed_document)
        
        if document_name is None:
            document_name = f"Stream Doc {len(self.stream_history) + 1}"
        self.document_names.append(document_name)
        
        # Vectorize ulang semua dokumen
        tfidf_vectorizer = TfidfVectorizer(
            min_df=self.base_analyzer.min_df,
            max_df=self.base_analyzer.max_df,
            ngram_range=self.base_analyzer.ngram_range
        )
        self.tfidf_matrix = tfidf_vectorizer.fit_transform(self.processed_documents)
        self.feature_names = np.array(tfidf_vectorizer.get_feature_names_out())
        
        # Menghitung kemiripan dengan dokumen lain
        new_doc_idx = len(self.documents) - 1
        similarities = {}
        
        for i, doc_name in enumerate(self.document_names[:-1]):
            sim = cosine_similarity(
                self.tfidf_matrix[i].reshape(1, -1),
                self.tfidf_matrix[new_doc_idx].reshape(1, -1)
            )[0][0]
            similarities[doc_name] = sim
        
        # Mendapatkan term penting dalam dokumen baru
        tfidf_scores = self.tfidf_matrix[new_doc_idx].toarray().flatten()
        top_indices = tfidf_scores.argsort()[-5:][::-1]
        top_terms = self.feature_names[top_indices]
        top_scores = tfidf_scores[top_indices]
        
        # Menyimpan hasil
        result = {
            'document_name': document_name,
            'top_terms': list(top_terms),
            'top_scores': list(top_scores),
            'similarities': similarities,
                            'most_similar_document': max(similarities.items(), key=lambda x: x[1])
            }
            
        self.stream_history.append(result)
            
        return result
        
    def get_stream_history(self):
        """
        Mendapatkan riwayat pemrosesan streaming.
        
        Returns:
        --------
        list
            Daftar hasil pemrosesan dokumen baru
        """
        return self.stream_history
    
    def visualize_stream_history(self):
        """
        Memvisualisasikan riwayat aliran dokumen.
        
        Returns:
        --------
        matplotlib.figure.Figure
            Figure hasil visualisasi
        """
        plt.figure(figsize=(12, 8))
        
        # Ekstrak data dari riwayat
        docs = [entry['document_name'] for entry in self.stream_history]
        similarities = [entry['most_similar_document'][1] for entry in self.stream_history]
        top_terms = [', '.join(entry['top_terms']) for entry in self.stream_history]
        
        # Buat plot
        plt.plot(docs, similarities, marker='o', linestyle='--', color='blue')
        plt.fill_between(docs, similarities, alpha=0.1, color='blue')
        
        # Tambahkan anotasi
        for i, (doc, sim, terms) in enumerate(zip(docs, similarities, top_terms)):
            plt.annotate(
                f"{sim:.2f}\n({terms})",
                (doc, sim),
                textcoords="offset points",
                xytext=(0,10),
                ha='center',
                fontsize=8
            )
        
        plt.title('Streaming Document Analysis History')
        plt.xlabel('Document Name')
        plt.ylabel('Maximum Similarity Score')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return plt.gcf()
    
if __name__ == "__main__":
    # Contoh dokumen dan nama dokumen
    documents = [
        "TF-IDF adalah metode statistik untuk mengukur pentingnya kata dalam dokumen.",
        "Text mining menggunakan TF-IDF untuk analisis dokumen.",
        "Machine learning adalah bagian dari kecerdasan buatan.",
        "Python populer untuk analisis data dan machine learning."
    ]
    
    document_names = [
        "Doc 1: TF-IDF",
        "Doc 2: Text Mining",
        "Doc 3: Machine Learning",
        "Doc 4: Python"
    ]

    # Inisialisasi basic analyzer
    basic_analyzer = TFIDFAnalyzer(
        language='indonesian',
        min_df=1,
        max_df=0.95,
        use_lemmatization=True
    )
    
    # Training dengan data
    basic_analyzer.fit(documents, document_names)
    
    # Analisis dasar
    top_terms = basic_analyzer.get_top_terms(n=5)
    similarity_matrix = basic_analyzer.calculate_document_similarity()
    clusters = basic_analyzer.cluster_documents(n_clusters=2)
    
    # Visualisasi basic
    basic_analyzer.visualize_clusters(method='pca')
    basic_analyzer.create_term_frequency_plot()
    plt.show()

    # Inisialisasi advanced analyzer
    advanced_analyzer = AdvancedTFIDFAnalyzer(
        language='indonesian',
        min_df=1,
        max_df=0.95,
        use_lemmatization=True
    )
    
    # Training dengan data yang sama
    advanced_analyzer.fit(documents, document_names)
    
    # Analisis lanjutan
    topics = advanced_analyzer.extract_topics(n_topics=2)
    sentiment = advanced_analyzer.analyze_sentiment()
    
    # Visualisasi advanced
    advanced_analyzer.visualize_topics()
    advanced_analyzer.visualize_sentiment(sentiment)
    plt.show()

    # Performance report
    print("\nPerformance Report:")
    print(basic_analyzer.create_performance_report())
    
    # Menyimpan model
    basic_analyzer.save_model("basic_tfidf_model.pkl")
    
    # Contoh streaming processing
    stream_processor = StreamingTFIDFProcessor(basic_analyzer)
    new_doc = "Natural Language Processing adalah bidang penting dalam AI"
    stream_result = stream_processor.process_new_document(new_doc, "Stream Doc 1")
    print("\nStreaming Processing Result:")
    print(stream_result)