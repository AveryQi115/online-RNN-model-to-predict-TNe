# USAGE
# python simple_request.py

# import the necessary packages
import requests

# initialize the Keras REST API endpoint URL along with the input
# image path
KERAS_REST_API_URL = "http://47.100.63.158/predict"
DATA_PATH = "test.txt"

# load the input image and construct the payload for the request
data = open(DATA_PATH, "r").read()

payload = {"data": data}

# submit the request
r = requests.post(KERAS_REST_API_URL, files=payload)
print(r)

# ensure the request was sucessful
#
#if r["success"]:
#	print(r['TNe'])

# otherwise, the request failed
#else:
#	print("Request failed")
