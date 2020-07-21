# import the necessary packages
import keras
from keras.models import load_model
import numpy as np
import flask
import pandas as pd
import tensorflow as tf
from flask import Flask, render_template, redirect, url_for


# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None

config = tf.ConfigProto(
		intra_op_parallelism_threads=1,
		allow_soft_placement=True
	)
session = tf.Session(config=config)

keras.backend.set_session(session)


def web_load_model():
	# load the trained model for TNe-hour
	global model
	path = 'TNe-hour.h5'
	model = load_model(path)

	# for solving the problem when load_model and model_predict through
	# web is not in the same thread
	testdata = np.zeros(shape=(1,5,8))
	pred = model.predict(testdata)
	#model.compile()
	#model.run_eagerly=True
	#print(model.input_shape)


def prepare_data(data, target):
	# resize the input data and preprocess it
	# the input data should be 5 x 8
	# 5 days of 'volume', 'CODi','NH3-Ni', 'TNi', 'TPi', 'CODe', 'NH3Ne', 'TPe'
	# todo: preprocessing
	data = data.reshape(target)
	data = np.expand_dims(data, axis=0)
	# print(data.shape)
	# return the processed data
	return data


@app.route('/')
def hello():
	return render_template("hello.html")


@app.route("/predict", methods=["GET","POST"])
def predict():
	# initialize the result dictionary that will be returned from the
	# view
	result = {"success": False}

	# ensure the data was properly uploaded to our endpoint
	if flask.request.method == "POST":
		if flask.request.files['data']:
			# read the data in tsv format
			data = pd.read_csv(flask.request.files["data"], sep='\t', header=None)
			data = np.array(data)
			# preprocess the data and prepare it for prediction
			data = prepare_data(data, target=(5, 8))

			try:
				with session.as_default():
					with session.graph.as_default():
						# classify the input data and then initialize the list
						# of predictions to return to the client
						preds = model.predict(data)
						result["TNe"] = str(preds.flatten()[0])

						# indicate that the request was a success
						result["success"] = True
			except Exception as ex:
				print('Seatbelt Prediction Error', ex)
		# return the data dictionary as a JSON response
		return render_template('result.html', result=result)
	elif flask.request.method == "GET":
		return render_template('predict.html')

@app.route("/about_us")
def about_us():
	return render_template("about_us.html")

@app.route("/about_program")
def about_program():
	return render_template("about_program.html")


# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
	print(("* Loading Keras model and Flask starting server..."
		"please wait until server has fully started"))
	web_load_model()
	app.run(host='0.0.0.0',port=5000)
	# app.run()
