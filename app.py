import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import joblib


carbodys=np.zeros(3)
drivewheels=np.zeros(2)
enginelocations=np.zeros(2)
enginetypes=np.zeros(2)
cylindernumbers=np.zeros(2)


app = Flask(__name__,static_folder='static')



@app.route('/')
def Home():
    return render_template("index.html")

@app.route("/predict", methods = ["POST","GET"])
def predict():
    
    carlength  =  float(request.args.get("carlength"))
    carwidth  =  float(request.args.get("carwidth"))
    curbweight  =  float(request.args.get("curbweight"))
    enginesize  =  float(request.args.get("enginesize"))
    horsepower = int(request.args.get("horsepower"))
    citympg  =  int(request.args.get("citympg"))
    carbody  =  int(request.args.get("carbody"))
    drivewheel  =  int(request.args.get("drivewheel"))
    enginelocation  =  int(request.args.get("enginelocation"))
    enginetype  =  int(request.args.get("enginetype"))
    cylindernumber  =  int(request.args.get("cylindernumber"))

    
  
    if carbody<=len(carbodys):
        carbodys[carbody-1]=1
    if drivewheel<=len(drivewheels):
        drivewheels[drivewheel-1]=1
    enginelocations[enginelocation-1]=1
    if enginetype<=len(enginetypes):
        enginetypes[enginetype-1]=1
    if cylindernumber<=len(cylindernumbers):    
        cylindernumbers[cylindernumber-1]=1
    
    
    carlength = np.array([carlength])
    carwidth = np.array([carwidth])
    curbweight = np.array([curbweight])
    enginesize = np.array([enginesize])
    horsepower = np.array([horsepower])
    citympg = np.array([citympg])
    
    
    model = pickle.load(open("D:/cv/car_price_prediction/model.pickle","rb"))
    with open("D:/cv/car_price_prediction/polynomial_regressor.pkl", 'rb') as file:
        loaded_regressor = pickle.load(file)

    carlength = carlength.flatten()
    carwidth = carwidth.flatten()
    curbweight = curbweight.flatten()
    enginesize = enginesize.flatten()
    horsepower = horsepower.flatten()
    citympg = citympg.flatten()

    features = np.concatenate((carlength, carwidth, curbweight, enginesize, horsepower, citympg, ) ).reshape(1, -1)
    
    features_en= np.concatenate(( carbodys, drivewheels, enginelocations, enginetypes, cylindernumbers)).reshape(1,-1)

    f_features=np.concatenate((features,features_en), axis=1).reshape(1,-1)
    


    prediction=float(model.predict(f_features))
    return render_template("index.html",prediction_text = "The predicted price is {}".format(prediction))
if __name__ == "__main__":
    app.run(debug=True)