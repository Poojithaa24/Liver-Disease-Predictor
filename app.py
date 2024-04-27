from flask import Flask,request, url_for, redirect, render_template, flash
import pickle
import numpy as np
from sklearn.metrics import accuracy_score
app = Flask(__name__)

LR = pickle.load(open('LR.pkl','rb'))
DT = pickle.load(open('DT.pkl','rb'))
RF = pickle.load(open('RF.pkl','rb'))
SVM = pickle.load(open('SVM.pkl','rb'))


@app.route('/')
def home():
    return render_template("liver_disease.html")

@app.route('/prediction')
def prediction():
    return render_template("instruction.html")

@app.route('/predict', methods=['POST', 'GET'])
def predict():

  all_features = list(request.form.values())
  int_features = [float(x) for x in all_features[:-1]]
  final = [np.array(int_features)]
  # Get the selected model from user input
  selected_model = request.form.get('model')
 
  # Use conditional statements to predict based on the selected model
  if selected_model == "Logistic Regression":
    prediction = LR.predict(final)
    
  elif selected_model == "Decision Trees":
    prediction = DT.predict(final)
    
  elif selected_model == "Randon Forest":
    prediction = RF.predict(final)

  elif selected_model == "SVM":
    prediction = SVM.predict(final)

  else:
    return render_template('instruction.html',pred='Not selected')
  if prediction==[1]:
    return render_template('instruction.html',pred='Your liver is in Danger.\n Seek medical health immediately.')
  else:
    return render_template('instruction.html',pred='Your liver is safe.\n Continue taking care.')

@app.route('/performance')
def performance():
  return render_template('performance.html')

@app.route('/code')
def code():
   return redirect("https://colab.research.google.com/drive/1sW_wmL_OCGRC8uS3eX3xJK3EgvJMeS87#scrollTo=lAjiCXh0zFrF")
if __name__ == '__main__':
    app.run(debug=True)
