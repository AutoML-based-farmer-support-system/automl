import pandas as pd
from h2o.automl import H2OAutoML
from h2o.estimators.gbm import H2OGradientBoostingEstimator
import h2o
from flask import Flask, request
from flask import jsonify
app = Flask(__name__)
h2o.init()
minpricepath = "minprice.zip"


@app.route('/')
def hello_world():
    return 'Hello Kisan'


@app.route('/train-minprice')
def minpricetraining():
    datasource = h2o.import_file(request.args.get("filepath"))
    datasource_train, datasource_test, datasource_valid = datasource.split_frame(
        ratios=[.7, .15])
    y = "min_price"
    x = datasource.columns
    aml = H2OAutoML(max_models=10, seed=10, verbosity="info", nfolds=0)
    aml.train(x=x, y=y, training_frame=datasource_train,
              validation_frame=datasource_valid)
    aml.leader.download_mojo(path="./minprice.zip")
    return "OK"
@app.route('/train-yield',methods = ['POST'])
def yieldtraining():
    datasource = h2o.import_file(request.form["filepath"])
    datasource_train, datasource_valid = datasource.split_frame(ratios = [.8], seed = 1234)
    y = "Yield"
    x = ['State_Name', 'District_Name','Crop']
    aml = H2OGradientBoostingEstimator(distribution = "poisson",seed=1234)
    aml.train(x=x, y=y, training_frame=datasource_train,
              validation_frame=datasource_valid)
    aml.download_mojo(path="./yield.zip")
    print("R2 value ")
    print(aml.r2())
    print("R2 validation value")
    print(aml.r2(valid=True))
    
    return "OK"    

@app.route('/train-maxprice')
def maxpricetraining():
    datasource = h2o.import_file(request.args.get("filepath"))
    datasource_train, datasource_test, datasource_valid = datasource.split_frame(
        ratios=[.7, .15])
    y = "max_price"
    x = datasource.columns
    aml = H2OAutoML(max_models=10, seed=10, verbosity="info", nfolds=0)
    aml.train(x=x, y=y, training_frame=datasource_train,
              validation_frame=datasource_valid)
    aml.leader.download_mojo(path="./maxprice.zip")
    return "OK"


@app.route('/predict-minprice')
def minpricepredict():
    minprice_model = h2o.import_mojo("D:\\automl\\minprice.zip")
    data = request.get_json()
    datalist = data['ma']
    df = pd.DataFrame(datalist, columns=[
                      'commodity_name', 'state', 'district', 'market'])
    hf = h2o.H2OFrame(df)
    predictredvalue = minprice_model.predict(hf)

    return jsonify({"predicteddata": h2o.as_list(predictredvalue, use_pandas=False)})

@app.route('/predict-yield')
def yieldpredict():
    yield_model = h2o.import_mojo("D:\\automl\\yield.zip")
    data = request.get_json()
    datalist = data['ma']
    df = pd.DataFrame(datalist, columns= ['State_Name', 'District_Name','Crop'])
    hf = h2o.H2OFrame(df)
    predictredvalue = yield_model.predict(hf)

    return jsonify({"predicteddata": h2o.as_list(predictredvalue, use_pandas=False)})    

@app.route('/predict-maxprice')
def maxpricepredict():
    maxprice_model = h2o.import_mojo("D:\\automl\\maxprice.zip")
    data = request.get_json()
    datalist = data['ma']
    df = pd.DataFrame(datalist, columns=[
                      'commodity_name', 'state', 'district', 'market'])
    hf = h2o.H2OFrame(df)
    predictredvalue = maxprice_model.predict(hf)

    return jsonify({"predicteddata": h2o.as_list(predictredvalue, use_pandas=False)})


if __name__ == '__main__':
    app.debug = False
    app.run()
