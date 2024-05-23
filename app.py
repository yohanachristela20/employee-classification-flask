import joblib
from flask import Flask, render_template, request
import numpy as np

app = Flask(__name__)

# Membuka file model dengan joblib
model = joblib.load("pegawai.joblib")

@app.route("/")
def main():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    # Mendapatkan nilai fitur dari permintaan POST
    JoiningYear = int(request.form['JoiningYear'])
    PaymentTier = int(request.form['PaymentTier'])
    Age = int(request.form['Age'])
    ExperienceInCurrentDomain = int(request.form['ExperienceInCurrentDomain'])
    Gender_Male = int(request.form['Gender_Male'])
    EverBenched_Yes = int(request.form['EverBenched_Yes'])
    Education_Masters = int(request.form['Education_Masters'])
    Education_PHD = int(request.form['Education_PHD'])
    City_New_Delhi = int(request.form['City_New_Delhi'])
    City_Pune = int(request.form['City_Pune'])

    # Melakukan prediksi dengan model
    prediction = model.predict([[JoiningYear, PaymentTier, Age, ExperienceInCurrentDomain, Gender_Male, EverBenched_Yes, Education_Masters, Education_PHD, City_New_Delhi, City_Pune]])

    # Menentukan teks prediksi berdasarkan hasil
    if prediction[0] == 1:
        employee_prediction = 'The employee is likely to leave'
    else:
        employee_prediction = 'The employee is not likely to leave'

    # Mengirim hasil prediksi ke template HTML
    return render_template("index.html", prediction_text=employee_prediction)

if __name__ == "__main__":
    app.run(debug=True)
