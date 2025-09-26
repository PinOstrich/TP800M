# app.py
from flask import Flask, render_template, request
from model import predict
from function import predict as TP500M

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        try:
            user_input = float(request.form["user_input"])
            result = TP500M(user_input)
        except ValueError:
            result = "Invalid input. Please enter a float."
    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)
