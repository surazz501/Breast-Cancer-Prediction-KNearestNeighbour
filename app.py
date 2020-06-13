import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
pickle_in = open('model.pickle','rb')
model = pickle.load(pickle_in)

@app.route('/')
def home():
    return render_template('index.htm')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
            thickness= int(request.form['thickness'])
            unif_cell_size = int(request.form['unif_cell_size'])
            unif_cell_shape = int(request.form['unif_cell_shape'])
            marge_adhesion= int(request.form['marge_adhesion'])
            single_epith_cell_size = int(request.form['single_epith_cell_size'])
            bare_nuclei = int(request.form['bare_nuclei'])
            bland_chrom = int(request.form['bland_chrom'])
            norm_nucleoli = int(request.form['norm_nucleoli'])
            mitoses = int(request.form['mitoses'])
            data = np.array([thickness,unif_cell_size,unif_cell_shape,marge_adhesion,single_epith_cell_size,bare_nuclei,bland_chrom,norm_nucleoli,mitoses])
            data = data.reshape(1,-1)
            my_prediction = model.predict(data)
            str_set = set(my_prediction)
            if 2 in str_set:
               my_prediction = 'Breast Cancer class is Benign'
            else:
               my_prediction= 'Breast Cancer class is Malignant'
            return render_template('index.htm', prediction_text=my_prediction)


if __name__ == "__main__":
    app.run(debug=True)
    