from flask import Flask, render_template, request, jsonify
import xmodel
from xmodel import get_headlines, make_prediction

app=Flask(__name__)

@app.route('/')
def base():
    return render_template('homepage.html')

@app.route('/news',methods=['GET', 'POST'])
def news():
  if request.method == 'POST':
    url = request.form['url']
    predict = xmodel.predict(url)
    value = predict[1]
    text = predict[2]
    article_title = predict[0]
    image=predict[3]
    return render_template('result.html',
                          value = value,
                          text = text,
                          article_title=article_title,
                          url=url,
                          image=image)
  else:
    return render_template('home.html')
  
@app.route('/hoax')
def hoax():
    return render_template('hoax.html')

@app.route('/news/api/v1/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
    except:
        return jsonify({"status": 503, "message": "an error occured"})
    data = request.get_json()
    label, predicted_label = make_prediction(data['text_narration'])
    prediction = {"status": 200, "text_narration": data['text_narration'], "prediction": {
       "label": label, "predicted_label": predicted_label}}
    return jsonify(prediction)

@app.route('/newsfeed',methods=['GET'])
def news_feed():
    headlines=get_headlines()

    return render_template('news_feed.html',headlines=headlines)

@app.route('/aboutus')
def about_us():
    return render_template('about_us.html')

if __name__=='__main__':
    app.run(port=8001, debug=True)