from . import *  
from app.irsystem.models.helpers import *
from app.irsystem.models.helpers import NumpyEncoder as NumpyEncoder
import json
from flask import Flask, request, redirect, g, render_template
import requests

project_name = "Ilan's Cool Project Template"
net_id = "Ilan Filonenko: if56"

TICKETMASTER_API_URL = "https://app.ticketmaster.com/discovery/v2/"
TICKETMASTER_API_KEY = "&apikey=TwBrYBbmHzChYbyzNgGYOk2NJVxKTNDs"

@irsystem.route('/', methods=['GET'])
def search():
	genre = request.args.get('genre')
	location = request.args.get('location')
	date = request.args.get('date')
	print(genre)
	if not (genre or location or date):
		print("blank")
		data = []
		output_message = ''
	else:
		output_message = "You're looking for " + genre + " concerts in " + location + " on " + date
		# data = range(5)
		search_endpoint = "{}events.json?classificationName=music&city={}&countryCode=US&startDateTime={}{}".format(TICKETMASTER_API_URL, location, date, TICKETMASTER_API_KEY)
		search_response = requests.get(search_endpoint)
		search_data = json.loads(search_response.text)
		data = search_data
		print(data)
	return render_template('search.html', name=project_name, netid=net_id, output_message=output_message, data=data)



