from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

ohe = pickle.load(open('models/ohe.pkl','rb'))
model = pickle.load(open('models/finalized_model.sav','rb'))

area_type = sorted(pickle.load(open('models/area.pkl','rb')))
location = sorted(pickle.load(open('models/location.pkl','rb')))

@app.route('/')
def index():
    return render_template("form.html",area_type= area_type,location = location)

@app.route('/predict',methods=['post'])
def predict():
    area = request.form.get('area')
    locat = request.form.get('location')
    total_sqft = request.form.get('total_sqft')
    BHK = request.form.get('BHK')
    bath = request.form.get('bath')
    balcony = request.form.get('balcony')

    X = np.array([BHK,total_sqft,bath,balcony],dtype=object).reshape(1,4)
    X_trans = np.array([[area,locat]],dtype=object).reshape(1,2)
    X_trans = ohe.transform(X_trans).toarray()

    X = np.asarray(np.hstack((X,X_trans)),dtype=object)
    y = model.predict(X)[0]
    y_ = "{:.2f}".format(y)
    y_ = "{:,}".format(int(float(y_)*10000))
    return render_template('form.html', original_input={'Area Type':area,'Location':locat,'Total Sqft':total_sqft,'BHK':BHK,'BATH':bath,'Balcony':balcony},result = y_)

if __name__ == "__main__":
    app.run(debug=True)
