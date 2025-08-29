from flask import Flask, render_template, request
from sklearn.linear_model import LogisticRegression
import pandas as pd


# Load data
url = "iris.csv"
iris = pd.read_csv(url)


# Features
x = iris[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values

# Target (string labels)
y = iris['species']


# Initialize and train the model
model = LogisticRegression(max_iter=200)
model.fit(x, y)

# prdiction
# pred = model.predict([[5.1, 3.5, 1.4, 0.2]])


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    sepal_length = float(request.form['sepal_length'])
    sepal_width = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])
    pred = model.predict(
        [[sepal_length, sepal_width, petal_length, petal_width]])
    return render_template('index.html', data=pred[0])


if __name__ == '__main__':
    app.run(debug=True)

