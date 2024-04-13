from flask import Flask,request, url_for, redirect, render_template, flash
import pickle
import numpy as np

app = Flask(__name__)

model=pickle.load(open('model.pkl','rb'))


@app.route('/')
def home():
    return render_template("liver_disease.html")

@app.route('/prediction')
def prediction():
    return render_template("instruction.html")


@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features=[float(x) for x in request.form.values()]
    final=[np.array(int_features)]
    prediction=model.predict(final)
    
    

    if prediction==[1]:
        return render_template('instruction.html',pred='Your liver is in Danger.\n Seek medical health immediately.')
    else:
        return render_template('instruction.html',pred='Your liver is safe.\n Continue taking care.')


if __name__ == '__main__':
    app.run(debug=True)
