#import necessary modules
from flask import Flask, request, render_template, jsonify
import pyforest
import pickle

#create flask app
app = Flask(__name__, template_folder='template')

#load model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def Home():
	return render_template('index.html')

@app.route('/predic', methods = ['POST'])
def predict():
	#convert independent values to float and save
	float_features = [float(x) for x in request.form.values()]
	#covert float_features to numpy arrays
	features = [np.array(float_features)]
	#make prediction
	prediction = model.predict(features)

	return render_template('index.html', prediction_test = "The Bank notes is {}        '1'- REAL NOTE '0' - FORGED NOTE".format(prediction))

if __name__ == '__main__':
	app.run(debug = True)