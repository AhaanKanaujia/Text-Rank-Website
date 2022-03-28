from flask import Flask, render_template, request
# from TextRank import return_keywords

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template('home.html')

@app.route('/keywords', methods=['POST'])
def keywords():
    text = request.form['text']
    keywords = return_keywords(text)
    return render_template('keywords.html', keywords = keywords, text = text)

if __name__ == "__main__":
    app.run()
