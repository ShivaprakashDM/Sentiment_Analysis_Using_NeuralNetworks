from flask import Flask, render_template, request
import pickle
import re
import nltk
import pandas as pd
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import warnings
from sklearn.exceptions import InconsistentVersionWarning

nltk.download('stopwords')
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

emoticon_pattern = re.compile(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)')

emojis = [':)', ':D']

stopwords_set = set(stopwords.words('english'))

app = Flask(__name__)

with open('clf.pkl', 'rb') as f:
    clf = pickle.load(f)
with open('tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)

def preprocessing(text):
    text = re.sub('<[^>]*>', '', text)
    emojis_found = emoticon_pattern.findall(text)
    text = re.sub(r'[\W+]', ' ', text.lower()) + ' '.join(emojis_found).replace('-', '')
    porter = PorterStemmer()
    text = [porter.stem(word) for word in text.split() if word not in stopwords_set]
    return " ".join(text)

data_path = r"C:\Users\DELL\Desktop\Sentiment-Analysis\Tweets.csv"
data = pd.read_csv(data_path, encoding='ISO-8859-1', header=None)

data.columns = ['textID','text','selected_text','sentiment']

print(data.columns)

data['text'] = data['text'].fillna('')  
data['cleaned_text'] = data['text'].apply(preprocessing)

tfidf_vectorizer = TfidfVectorizer()

X = tfidf.transform(data['cleaned_text'])

num_clusters = 4 
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
data['cluster'] = kmeans.fit_predict(X)

@app.route('/', methods=['GET', 'POST'])
def analyze_sentiment():
    if request.method == 'POST':
        comment = request.form.get('comment')
        
        preprocessed_comment = preprocessing(comment)

        comment_vector = tfidf.transform([preprocessed_comment])

        prediction_prob = clf.predict_proba(comment_vector)[0]
        
        if prediction_prob[1] >= 0.7:
            sentiment = 1  # Positive sentiment
        elif prediction_prob[0] >= 0.7:
            sentiment = 0  # Negative sentiment
        else:
            sentiment = 2  # Neutral sentiment

        cluster = kmeans.predict(comment_vector)[0]

        return render_template('index.html', sentiment=sentiment, cluster=cluster)

    return render_template('index.html')


@app.route('/cluster', methods=['POST'])
def cluster_comments():
    comment = request.form.get('comment')

    preprocessed_comment = preprocessing(comment)

    comment_vector = tfidf_vectorizer.transform([preprocessed_comment])

    cluster = kmeans.predict(comment_vector)

    return render_template('index.html', cluster=cluster[0])

if __name__ == '__main__':
    app.run(debug=True)
