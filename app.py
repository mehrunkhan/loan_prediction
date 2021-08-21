# save this as app.py
from flask import Flask, escape, request, render_template
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('data/decision_tree_model.sav', 'rb'))


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/about_us')
def home():
    return render_template("about_us.html")


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':

        code_gender = int(request.form['code_gender'])
        flag_own_car = int(request.form['flag_own_car'])
        flag_own_realty = int(request.form['flag_own_realty'])
        cnt_children = int(request.form['cnt_children'])
        amt_income_total = float(request.form['amt_income_total'])
        desired_credit = float(request.form['desired_credit'])
        name_education_type = int(request.form['name_education_type'])
        name_family_status = request.form['name_family_status']
        name_housing_type = request.form['name_housing_type']
        days_birth = int(request.form['days_birth'])
        flag_work_phone = int(request.form['flag_work_phone'])
        flag_cont_mobile = int(request.form['flag_cont_mobile'])
        occupation_type = request.form['occupation_type']
        cnt_fam_members = int(request.form['cnt_fam_members'])
        reg_region_not_work_region = int(request.form['reg_region_not_work_region'])
        days_last_phone_change = float(request.form['days_last_phone_change'])
        document_passport = int(request.form['document_passport'])
        document_visa_or_resident_permit = int(request.form['document_visa_or_resident_permit'])
        document_tax_identification_number = int(request.form['document_tax_identification_number'])
        document_city_registration = int(request.form['document_city_registration'])
        document_matriculation_number = int(request.form['document_matriculation_number'])
        document_social_security_number = int(request.form['document_social_security_number'])

        prediction = model.predict([[code_gender, flag_own_car, flag_own_realty, cnt_children, amt_income_total,
                                     desired_credit, name_education_type, name_family_status, name_housing_type,
                                     -days_birth / 365, flag_work_phone,
                                     flag_cont_mobile, occupation_type, cnt_fam_members, reg_region_not_work_region,
                                     -days_last_phone_change, document_passport, document_visa_or_resident_permit,
                                     document_tax_identification_number, document_city_registration,
                                     document_matriculation_number, document_social_security_number]])

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
