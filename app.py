import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)
model = joblib.load('models/linear_regression.pkl')

@app.route("/")
def home():
    return render_template('index.html')

#### Linear Regression
@app.route("/linear_regression/")
def linreg():
    return render_template('index_linreg.html')

@app.route('/linear_regression/predict/',methods=['POST'])
def predict():

    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)

    output = float(prediction[0])
    output = round(output, 2)

    return render_template('index_linreg.html', prediction_text='Price should be $ {}'.format(output))

@app.route('/linear_regression/results/',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

#### Multiple Linear Regression

if __name__ == "__main__":
    app.run(debug=True)