from flask import Flask, app, render_template, request
from logging import debug
from flask.templating import render_template
import nltk
import string
import pandas as pd
import pycountry
from textblob import TextBlob
from textblob import Word

app  = Flask(__name__)

@app.route("/")
def form():
    return render_template('home.html')

@app.route("/count", methods=['POST'])
def count():
    return render_template('word_count_nlp.html')

@app.route("/count_result", methods=['POST'])
def count_result():

    data = request.form.get('user_message')
    
    nltk.download('punkt')
    # Splitting the data to  
    words = nltk.word_tokenize(data)
    sentences = nltk.sent_tokenize(data)
    punct = string.punctuation
    new_words = []
    for i in words:
        if i not in punct:
            new_words.append(i)   
                
    word = len(new_words)
    sent = len(sentences)
    
    return render_template('word_count_nlp.html', prediction_txt = f"Number of words in the data is: {word} and number of sentences in data is: {sent}")

@app.route("/dictionary", methods=['POST'])
def dictionary():
    return render_template('dic_nlp.html')

@app.route("/dictionary_result", methods=['POST'])
def dictionary_result():

    nltk.download('punkt')

    data = request.form.get('dic_text')
    meaning = Word(data).definitions
    return render_template('dic_nlp.html', txt_definition = f"{meaning}")

@app.route("/lang_detect", methods=['POST'])
def lang_detect():
    return render_template('lang_nlp.html')

@app.route("/lang_detect_result", methods=['POST'])
def lang_detect_result():

    nltk.download('punkt')

    data = request.form.get('user_text')
    blob = TextBlob(data)
    iso_code = blob.detect_language()
    language = pycountry.languages.get(alpha_2=iso_code)
    language_name = language.name

    return render_template('lang_nlp.html', detect_lang = f"The entered Text is in: {language_name}")

@app.route("/translate", methods=['POST'])
def translate():
    return render_template('translate_nlp.html')

@app.route("/translate_result", methods=['POST'])
def translate_result():

    nltk.download('punkt')

    codes = pd.read_csv('lang_codes.csv')
    codes.columns = ['Language', 'Code']

    txt = request.form.get('user_text')
    lang = request.form.get('dropdown')
    x = codes[codes.Language == lang].Code.values[0]
    blob = TextBlob(txt)
    return render_template('translate_nlp.html', translated_txt = f"{blob.translate(to = x)}")

@app.route("/check_spell", methods=['POST'])
def check_spell():
    return render_template('spell_check_nlp.html')

@app.route("/check_spell_result", methods=['POST'])
def check_spell_result():

    nltk.download('punkt')

    data = request.form.get('user_text')
    k = TextBlob(data)
    crct_spell = k.correct()
    return render_template('spell_check_nlp.html', correct_spell = f"{crct_spell}")

if __name__ == "__main__":
    app.run(debug=True)
