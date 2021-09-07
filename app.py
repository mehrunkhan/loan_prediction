# save this as app.py
from flask import Flask, escape, request, render_template
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('trained_models/forest_model.sav', 'rb'))


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/about_us')
def about_us():
    return render_template("about_us.html")


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':

        Gender = int(request.form['Gender'])
        OwnCar = int(request.form['OwnCar'])
        OwnRealty = int(request.form['OwnRealty'])
        ChildrenCount = int(request.form['ChildrenCount'])
        TotalIncome = float(request.form['TotalIncome'])
        DesiredCredit = float(request.form['DesiredCredit'])
        HighestEducation = int(request.form['HighestEducation'])
        FamilyStatus = request.form['FamilyStatus']
        HousingType = request.form['HousingType']
        DaysBirth = int(request.form['DaysBirth'])
        WorkPhone = int(request.form['WorkPhone'])
        ContractMobile = int(request.form['ContractMobile'])
        Occupation = request.form['Occupation']
        FamilyMembersCount = int(request.form['FamilyMembersCount'])
        DaysLastPhoneChange = float(request.form['DaysLastPhoneChange'])
        Passport = int(request.form['Passport'])
        TaxIdentification = int(request.form['TaxIdentification'])
        CityRegistration = int(request.form['CityRegistration'])
        MatriculationNumber = int(request.form['MatriculationNumber'])
        SocialSecurity = int(request.form['SocialSecurity'])
        VisaResidentPermit = int(request.form['VisaResidentPermit'])

        prediction = model.predict([[Gender, OwnCar, OwnRealty, ChildrenCount, TotalIncome,
                                     DesiredCredit, HighestEducation, FamilyStatus, HousingType,
                                     -DaysBirth / 365, WorkPhone,
                                     ContractMobile, Occupation, FamilyMembersCount,
                                     -DaysLastPhoneChange, Passport,
                                     TaxIdentification, CityRegistration,
                                     MatriculationNumber, SocialSecurity, VisaResidentPermit]])

        print(prediction)

        if prediction == 0:
            prediction = "Rejected"
        else:
            prediction = "Approved"

        return render_template("prediction.html", prediction_text="loan status is {}".format(prediction))




    else:
        return render_template("prediction.html")


if __name__ == "__main__":
    app.run(debug=True)
