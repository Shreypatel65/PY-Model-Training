import uuid
import pickle
import pandas as pd
from datetime import datetime
from pymongo import MongoClient
from django.shortcuts import render,redirect
from sklearn.linear_model import LinearRegression

client = MongoClient('mongodb://127.0.0.1:27017/?directConnection=true&serverSelectionTimeoutMS=2000&appName=mongosh+1.10.1')
db = client['py']
tmpcollection = db['temp_data']
model_collection = db['models']

def index(request):
    # Retrieve models from MongoDB
    models = list(model_collection.find({}))
    return render(request,'index.html', {'models':models})

def review_model(request):
    # Upload and review CSV data
    file = request.FILES.get('data_file')
    data = pd.read_csv(file)
    uid = str(uuid.uuid4())
    data_dict = data.to_dict(orient='records')
    tmpcollection.insert_one({'_id': uid, 'data': data_dict})
    models = list(model_collection.find({}))
    return render(request,'index.html', {'data':data, 'uid':uid, 'models':models})

def train_model(request):
    # Train a linear regression model
    uid = request.POST['uuid']
    data_columns = request.POST.getlist('data_columns')
    pred_columns = request.POST.getlist('pred_columns')
    model_name = request.POST['model_name']

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
    model_collection.insert_one({'model_name': model_name, 'create_date': date, 'data_columns': list(X.columns), 'pred_columns': pred_columns, 'model_data': binary_model})

    return redirect('index')
def input_output(request,model_name):
    if request.method == 'GET':
        # Get input data for making predictions
        modellst = model_collection.find({'model_name': model_name})
        model = pickle.loads(modellst[0]['model_data'])
        pred = {item: None for item in modellst[0]['pred_columns']}
        data = {item: None for item in modellst[0]['data_columns']}
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
                
        output = {
            'model_name':model_name,
            'pred_data':pred,
            'input_data':data,
            'create_date':modellst[0]['create_date']
        }
        
        return render(request,'predict.html', {'models':output, 'eq':equation})

    else:
        # Predict output based on input data
        input_data = request.POST
        equation = eval(request.POST['eq'])
        data_columns = eval(request.POST['data_columns'])
        pred_columns = eval(request.POST['pred_columns'])

        input_values = [{i: float(input_data[i]) for i in data_columns}]

        list_model = list(model_collection.find({'model_name': input_data['model_name']}))
        model = pickle.loads(list_model[0]['model_data'])

        input_df = pd.DataFrame(input_values)
        predicted_result = model.predict(input_df)
        
        try:
            if (len(predicted_result[0])):
                temp = [predicted_result][0][0]
        except:
            temp = [predicted_result][0]
        
        pred_data = {i: j for i, j in zip(pred_columns, temp)}
        output = {
            'model_name': input_data['model_name'],
            'pred_data':pred_data,
            'input_data':input_values[0]
            }
        
        return render(request,'predict.html', {'models':output, 'eq':equation})

