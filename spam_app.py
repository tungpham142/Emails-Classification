from flask import Flask, render_template, url_for, request
from spam_filter import spam_filter
import nltk

app = Flask(__name__)
spam_filter = spam_filter()
result = []

@app.route("/")
def home():
	return render_template('index.html', data = spam_filter.data)

@app.route("/", methods=['POST'])
def process():
	classify = request.form.get('classify')
	if(classify != None):
		classification = spam_filter.classify(classify)		
		return render_template('index.html', classification = classification)

if __name__ == '__main__':
	app.run(debug=True)