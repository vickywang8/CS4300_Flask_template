from . import *  
from app.irsystem.models.helpers import *
from app.irsystem.models.helpers import NumpyEncoder as NumpyEncoder
import json
from flask import Flask, request, redirect, g, render_template
import requests
import json

project_name = "Ilan's Cool Project Template"
net_id = "Ilan Filonenko: if56"

TICKETMASTER_API_URL = "https://app.ticketmaster.com/discovery/v2/"
TICKETMASTER_API_KEY = "&apikey=TwBrYBbmHzChYbyzNgGYOk2NJVxKTNDs"

@irsystem.route('/', methods=['GET'])
def search():
	queried_genre = request.args.get('genre')
	queried_location = request.args.get('location')
	queried_date = request.args.get('date')
	if not (queried_genre or queried_location or queried_date):
		print("blank")
		data = []
		output_message = ''
	else:
		output_message = "You're looking for " + queried_genre + " concerts in " + queried_location + " on " + queried_date
		# data = range(5)
		search_endpoint = "{}events.json?classificationName=music&city={}&countryCode=US&startDateTime={}{}".format(TICKETMASTER_API_URL, queried_location, queried_date, TICKETMASTER_API_KEY)
		search_response = requests.get(search_endpoint)
		search_data = json.loads(search_response.text)
		data = search_data
		events = data["_embedded"]["events"]
		concerts_dict = {}
		for event in events:
			genre = event["classifications"][0]["genre"]["name"]
			if queried_genre.lower() in genre.lower():
				name = event["name"]
				date = event["dates"]["start"]["localDate"]
				try:
					time = event["dates"]["start"]["localTime"]
				except:
					time = ""
				link = event["url"]
				location = event["_embedded"]["venues"][0]["name"]
				concert = {"date": date, "time": time, "link": link, "location": location}
				if name not in concerts_dict:
					concerts_dict[name] = [concert]
				else:
					concerts_dict[name].append(concert)

		for concert in concerts_dict.keys():
			print concert
	return render_template('search.html', name=project_name, netid=net_id, output_message=output_message, data=data)



