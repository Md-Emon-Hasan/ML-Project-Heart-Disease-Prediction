from flask import Flask
from flask import request
from flask import render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load the model and data
data = pickle.load(open('./models/data.pkl', 'rb'))
pipe = pickle.load(open('./models/model.pkl', 'rb'))

@app.route('/', methods=['GET'])
def index():
    # Sort the unique values
    sex_sort = sorted(data['sex'].unique())
    cp_sort = sorted(data['cp'].unique())
    fbs_sort = sorted(data['fbs'].unique())
    restecg_sort = sorted(data['restecg'].unique())
    exang_sort = sorted(data['exang'].unique())
    slope_sort = sorted(data['slope'].unique())
    ca_sort = sorted(data['ca'].unique())
    thal_sort = sorted(data['thal'].unique())
    
    return render_template('index.html', sexs=sex_sort, cps=cp_sort, fbss=fbs_sort, restecgs=restecg_sort, exangs=exang_sort, slopes=slope_sort, cas=ca_sort, thals=thal_sort)

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve form data
    age = int(request.form['age'])
    sex = int(request.form['sex'])
    cp = int(request.form['cp'])
    trestbps = int(request.form['trestbps'])
    chol = int(request.form['chol'])
    fbs = int(request.form['fbs'])
    restecg = int(request.form['restecg'])
    thalach = int(request.form['thalach'])
    exang = int(request.form['exang'])
    oldpeak = float(request.form['oldpeak'])
    slope = int(request.form['slope'])
    ca = int(request.form['ca'])
    thal = int(request.form['thal'])
    
    # Create a DataFrame from the input data for the model
    query = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]], columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'])
    
    # Predict survival
    prediction = pipe.predict(query)[0]
    
    # Sort unique values again for dropdown options
    sex_sort = sorted(data['sex'].unique())
    cp_sort = sorted(data['cp'].unique())
    fbs_sort = sorted(data['fbs'].unique())
    restecg_sort = sorted(data['restecg'].unique())
    exang_sort = sorted(data['exang'].unique())
    slope_sort = sorted(data['slope'].unique())
    ca_sort = sorted(data['ca'].unique())
    thal_sort = sorted(data['thal'].unique())
    
    # Render template with prediction and input values
    return render_template(
        'index.html',
        prediction=prediction,
        sexs=sex_sort,
        cps=cp_sort,
        fbss=fbs_sort,
        restecgs=restecg_sort,
        exangs=exang_sort,
        slopes=slope_sort,
        cas=ca_sort,
        thals=thal_sort, 
        age=age,
        sex=sex,
        cp=cp,
        trestbps=trestbps,
        chol=chol,
        fbs=fbs,
        restecg=restecg,
        thalach=thalach,
        exang=exang,
        oldpeak=oldpeak,
        slope=slope,
        ca=ca,
        thal=thal
    )

if __name__ == '__main__':
    app.run(
        debug=True,
        host='0.0.0.0',
        port=5000)
