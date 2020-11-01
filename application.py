from flask import Flask ,render_template, request
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/predict', methods=['POST'])
def predict():
    
        int_features = [int(x) for x in request.form.values()]
        arr_features = [np.array(int_features)]
        prediction = model.predict(arr_features)
        result=round(prediction[0],1)
    
        return render_template('index1.html', prediction_text='Employee Salary should be INR {}'.format(result)) 
    
if __name__== "__main__":
    app.run(debug=True)