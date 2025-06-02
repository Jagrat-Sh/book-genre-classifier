from flask import Flask, render_template, request
from transformers import pipeline

app = Flask(__name__)
classifier = pipeline("zero-shot-classification", model="sshleifer/tiny-distilroberta-base")

CANDIDATE_GENRES = [
    "science fiction", "romance", "thriller", "fantasy",
    "historical", "non-fiction", "mystery", "horror", "young adult"
]

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        blurb = request.form["blurb"]
        prediction = classifier(blurb, CANDIDATE_GENRES)
        result = list(zip(prediction["labels"], prediction["scores"]))
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
