from newspaper import Article
import pandas as pd
import numpy as np
import pickle, requests, re, datetime
from newsapi import NewsApiClient

newsapi=NewsApiClient(api_key='d02626a2008a45539c7e551b4b87c842')

passiveaggressive_model=pickle.load(open("modelNewIndo.pickle","rb"))
tfidf_vectorizer=pickle.load(open("tfidfNewIndo.pickle","rb"))

def predict_fake(title,text):
    
    data={"Unnamed: 0":["0000"], "title":[title], "text":[text], "label":["FAKE/REAL"]}
    frame=pd.DataFrame(data, columns= ["Unnamed: 0", "title","text","label"])
    frame.drop("label",axis=1)
    tfidf_test=tfidf_vectorizer.transform(frame['text'])
    pred = passiveaggressive_model.predict(tfidf_test)
    if pred[0] == 0:
        return "Berita Asli"
    else:
        return "Berita Palsu"

def predict(url):
    
    try:
        article=Article(url, language='id')
        article.download()
        article.parse()
        
        if len(article.text)<=500:
            return[str(article.title)]+(["INVALID"]*3)

        article.nlp()
        
        return [str(article.title), predict_fake(str(article.title),str(article.text)),str(article.summary),article.top_image]
    except ValueError:
        return(["INVALID"]*4)
    finally:
        if len(article.text)<=500:
            return[str(article.title)] + (["INVALID"]*4)
        return  [str(article.title), predict_fake(str(article.title), str(article.text)),  str(article.summary),article.top_image] 

def get_headlines():
    final = []
    url = "https://newsapi.org/v2/top-headlines"
    params = {
        "country": "id",
        "apiKey": "d02626a2008a45539c7e551b4b87c842",
        "pageSize": "9"
    }

    response = requests.get(url, params=params)
    data = response.json()

    # Cek status respon
    if response.status_code == 200:
        # Ambil hasil berita
        articles = data["articles"]

        # Loop melalui setiap artikel
        for article in articles:
            title = article["title"]
            source = article["source"]["name"]
            article_url = article["url"]
            parsed_date = datetime.datetime.strptime(article["publishedAt"], "%Y-%m-%dT%H:%M:%SZ")
            formatted_date = parsed_date.strftime("%d-%m-%Y")
            # Buat objek Article dari URL
            news_article = Article(article_url)
            news_article.download()
            news_article.parse()
            news_article.nlp()
            description = news_article.summary

            # Prediksi label berita
            prediction = predict(article_url)
            label = "Berita Palsu" if prediction == 1 else "Berita Asli"

            # Tambahkan data ke dalam list final
            final.append([article_url, title, description, source, news_article.top_image, label, formatted_date])

    else:
        print("Gagal mendapatkan berita. Kode status:", response.status_code)

    return final

def make_prediction(text_narration):
    """
    Make a prediction from a string
    return logits, label
    """
    text = text_normalization(text_narration)
    tfidf_text=tfidf_vectorizer.transform([text])
    label = passiveaggressive_model.predict(tfidf_text)
    decision_scores = passiveaggressive_model.decision_function(tfidf_text)
    prob = 1 / (1 + np.exp(-decision_scores))
    return label.tolist(), prob.tolist()

def text_normalization(text_narration):
    """
    This method normalize text provided by users
    """
    text_narration = text_narration.lower()
    return text_narration