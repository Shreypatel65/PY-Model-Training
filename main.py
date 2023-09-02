import uuid
import pickle
import pandas as pd
from datetime import datetime
from pymongo import MongoClient
from sklearn.linear_model import LinearRegression
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

# Connect to MongoDB
client = MongoClient('mongodb://127.0.0.1:27017/?directConnection=true&serverSelectionTimeoutMS=2000&appName=mongosh+1.10.1')
db = client['py']
tmpcollection = db['temp_data']
model_collection = db['models']

@app.route('/')
def index():
    # Retrieve models from MongoDB
    models = list(model_collection.find({}))
    return render_template('index.html', models=models)

@app.route('/review', methods=['POST'])
def review_model():
    # Upload and review CSV data
    file = request.files['data_file']
    data = pd.read_csv(file)
    uid = str(uuid.uuid4())
    data_dict = data.to_dict(orient='records')
    tmpcollection.insert_one({'_id': uid, 'data': data_dict})
    models = list(model_collection.find({}))
    return render_template('index.html', data=data, uid=uid, models=models)

@app.route('/train', methods=['POST'])
def train_model():
    # Train a linear regression model
    uid = request.form['uuid']
    data_columns = request.form.getlist('data_columns')
    pred_columns = request.form.getlist('pred_columns')
    model_name = request.form['model_name']

    processed_data = tmpcollection.find_one({'_id': uid})

    if processed_data is None:
        return "Processed data not available"

    data = pd.DataFrame.from_dict(processed_data['data'])
    data.dropna(inplace=True)
    X = data[data_columns]
    Y = data[pred_columns[0]]
    
    for i in data_columns:
        if X[i].dtypes == 'object':
            X = pd.get_dummies(X, columns=[i])
    
    if Y.dtype == 'object':
        Y = pd.get_dummies(Y)
        pred_columns = list(Y.columns)
    
    model = LinearRegression()
    model.fit(X, Y)
    binary_model = pickle.dumps(model)

    date = datetime.now()
    model_collection.insert_one(
        {'model_name': model_name, 'create_date': date, 'data_columns': list(X.columns), 'pred_columns': pred_columns, 'model_data': binary_model})

    return redirect(url_for('index'))

@app.route('/predict_result', methods=['GET'])
def get_input():
    # Get input data for making predictions
    model_name = request.args.get('model_name')
    modellst = model_collection.find({'model_name': model_name})
    model = pickle.loads(modellst[0]['model_data'])
    pred = modellst[0]['pred_columns']
    data = modellst[0]['data_columns']
    equation = []

    if len(pred) == 1 and len(data) == 1:
        equation = [
            f'{pred}: y = {model.coef_:.4f}{data} + {model.intercept_:.4f}\n']
    elif len(pred) == 1:
        for i, ycol in enumerate(pred):
            coefficients = list(model.coef_)
            intercept = model.intercept_
            equation_parts = []
            for feature, coef in zip(data, coefficients):
                equation_parts.append(f'{coef:.4f} * {feature}')
            equation.append(
                f'{ycol}: y = {" + ".join(equation_parts)} + {intercept:.4f}\n')
    else:
        for i, ycol in enumerate(pred):
            coefficients = model.coef_[i]
            intercept = model.intercept_[i]

            equation_parts = []
            for feature, coef in zip(data, coefficients):
                equation_parts.append(f'({coef:.4f} * {feature})')
            equation.append(
                f'{ycol}: y = {"+".join(equation_parts)} +{intercept:.4f}\n')
    
    return render_template('predict.html', models=modellst, eq=equation)

@app.route('/predict_result', methods=['POST'])
def predict_output():
    # Predict output based on input data
    input_data = request.form
    equation = eval(input_data['eq'])
    data_columns = eval(input_data['data_columns'])
    pred_columns = eval(input_data['pred_columns'])

    new_data = [{i: float(input_data[i]) for i in data_columns}]

    list_model = list(model_collection.find(
        {'model_name': input_data['model_name']}))
    model = pickle.loads(list_model[0]['model_data'])

    input_df = pd.DataFrame(new_data)
    predicted_result = model.predict(input_df)
    
    try:
        if (len(predicted_result[0])):
            temp = [predicted_result][0][0]
    except:
        temp = [predicted_result][0]
    
    pred_data = {i: j for i, j in zip(pred_columns, temp)}
    output = {'pred_columns': pred_columns, 'data_columns': data_columns,
              'model_name': input_data['model_name']}
    output.update(pred_data)
    output.update(new_data[0])
    
    return render_template('predict.html', models=[output], eq=equation)

app.run(debug=True)
