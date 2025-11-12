from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load model
with open("LR.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get input values from form
    sepal_length = float(request.form["sepal_length"])
    sepal_width = float(request.form["sepal_width"])
    petal_length = float(request.form["petal_length"])
    petal_width = float(request.form["petal_width"])

    # Predict directly
    pred = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    flower_name = pred[0]  # model already gives the name

    return render_template("index.html", result=flower_name)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
