from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load the model
data = pickle.load(open('./model/raw_data.pkl', 'rb'))
model = pickle.load(open('./model/model.pkl', 'rb'))

@app.route('/', methods=['GET'])
def index():
    # Sort the unique values
    source_sort = sorted(data['source'].unique())
    browser_sort = sorted(data['browser'].unique())
    sex_sort = sorted(data['sex'].unique())
    country_name_sort = sorted(data['country_name'].unique())
    signup_day_name_sort = sorted(data['signup_day_name'].unique())
    purchase_day_name_sort = sorted(data['purchase_day_name'].unique())

    return render_template(
        'index.html', 
        sources=source_sort, 
        browsers=browser_sort, 
        sexs=sex_sort, 
        country_names=country_name_sort, 
        signup_day_names=signup_day_name_sort, 
        purchase_day_names=purchase_day_name_sort
    )

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve form data
    source = request.form['source']
    browser = request.form['browser']
    sex = request.form['sex']
    age = int(request.form['age'])
    country_name = request.form['country_name']
    n_device_occur = int(request.form['n_device_occur'])
    signup_month = int(request.form['signup_month'])
    signup_day = int(request.form['signup_day'])
    signup_day_name = request.form['signup_day_name']
    purchase_month = int(request.form['purchase_month'])
    purchase_day = int(request.form['purchase_day'])
    purchase_day_name = request.form['purchase_day_name']
    purchase_over_time = float(request.form['purchase_over_time'])

    # Create a DataFrame from the input data for the model
    query = pd.DataFrame([[source, browser, sex, age, country_name, n_device_occur, signup_month, signup_day, signup_day_name, purchase_month, purchase_day, purchase_day_name, purchase_over_time]], columns=['source', 'browser', 'sex', 'age','country_name', 'n_device_occur', 'signup_month', 'signup_day', 'signup_day_name', 'purchase_month', 'purchase_day', 'purchase_day_name', 'purchase_over_time'])

    # Predict
    prediction = model.predict(query)[0]

    # Sort unique values again for dropdown options
    source_sort = sorted(data['source'].unique())
    browser_sort = sorted(data['browser'].unique())
    sex_sort = sorted(data['sex'].unique())
    country_name_sort = sorted(data['country_name'].unique())
    signup_day_name_sort = sorted(data['signup_day_name'].unique())
    purchase_day_name_sort = sorted(data['purchase_day_name'].unique())

    # Render template with prediction and input values
    return render_template(
        'index.html',
        prediction=prediction,
        sources=source_sort,
        browsers=browser_sort,
        sexs=sex_sort,
        country_names=country_name_sort,
        signup_day_names=signup_day_name_sort,
        purchase_day_names=purchase_day_name_sort,
        source=source,
        browser=browser,
        sex=sex,
        age=age,
        country_name=country_name,
        n_device_occur=n_device_occur,
        signup_month=signup_month,
        signup_day=signup_day,
        signup_day_name=signup_day_name,
        purchase_month=purchase_month,
        purchase_day=purchase_day,
        purchase_day_name=purchase_day_name,
        purchase_over_time=purchase_over_time
    )

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)