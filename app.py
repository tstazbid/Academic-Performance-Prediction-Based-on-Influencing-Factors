from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def hello_world():
    return render_template("home.html")


@app.route('/predict', methods=['POST'])
def predict():
    print("Predict function is called")

    int_features = [int(x) for x in request.form.values()]
    print("Form values:", int_features)

    final = [np.array(int_features)]
    print("Final array:", final)

    predicted_value = model.predict(final)
    print("Predicted value:", predicted_value)
    
    if predicted_value < 0.5:
        got_output = 0
    else:
        got_output = 1
        
    if got_output == 0:
        return render_template('home.html', pred='Grade will be poor')
    else:
        return render_template('home.html', pred='Grade will be high')


if __name__ == '__main__':
    app.run(debug=True)
    
