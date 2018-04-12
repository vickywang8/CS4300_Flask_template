from . import *  
from app.irsystem.models.helpers import *
from app.irsystem.models.helpers import NumpyEncoder as NumpyEncoder
import json
from flask import Flask, request, redirect, g, render_template
import requests
import json
import dateutil.parser as dparser
import datetime as dt


project_name = "Ilan's Cool Project Template"
net_id = "Ilan Filonenko: if56"

TICKETMASTER_API_URL = "https://app.ticketmaster.com/discovery/v2/"
TICKETMASTER_API_KEY = "&apikey=TwBrYBbmHzChYbyzNgGYOk2NJVxKTNDs"

@irsystem.route('/', methods=['GET'])
def search():
	queried_artist = request.args.get('artist')
	queried_genre = request.args.get('genre')
	queried_location = request.args.get('location')
	queried_date = request.args.get('date')
	data = None
	if not (queried_artist or queried_genre or queried_date or queried_location):
		data = []
		output_message = ""
	else:
		if not queried_genre:
			queried_genre = "music"
		if queried_date:
			start_date, end_date = format_date(queried_date)
		# output_message = format_output_message(queried_genre, queried_location, queried_date)
		output_message = ""
		search_endpoint = "{}events.json?classificationName={}&city={}&countryCode=US&startDateTime={}&endDateTime={}&keyword={}&includeSpellcheck=yes{}".format(TICKETMASTER_API_URL, queried_genre, queried_location, start_date, end_date, queried_artist, TICKETMASTER_API_KEY)
		try:
			search_response = requests.get(search_endpoint)
			search_data = json.loads(search_response.text)
			if "spellcheck" in search_data.keys():
				print(search_data)
				spellchecked_artist = search_data["spellcheck"]["suggestions"][0]["suggestion"]
				search_endpoint = "{}events.json?classificationName={}&city={}&countryCode=US&startDateTime={}&endDateTime={}&keyword={}&includeSpellcheck=yes{}".format(TICKETMASTER_API_URL, queried_genre, queried_location, start_date, end_date, spellchecked_artist, TICKETMASTER_API_KEY)
				search_response = requests.get(search_endpoint)
				search_data = json.loads(search_response.text)	
			data = get_concert_data(search_data)	
		except Exception as e:
			date = []
			output_message = "Your search returned no results. Modify your search and try again!"
			print(e)
	return render_template('search.html', name=project_name, netid=net_id, output_message=output_message, data=data, queried_genre= queried_genre, queried_date=queried_date, queried_location=queried_location, queried_artist=queried_artist)

def get_concert_data(search_data):
	events = search_data["_embedded"]["events"]
	concerts = []
	for event in events:
		name = event["name"]
		date = get_readable_date(event["dates"]["start"]["localDate"])
		try:
			time = get_readable_time(event["dates"]["start"]["localTime"])
		except:
			time = ""
		link = event["url"]
		location = event["_embedded"]["venues"][0]["name"]
		concert_data = {"name": name, "date": date, "time": time, "link": link, "location": location}
		concerts.append(concert_data)
	return concerts

## date range picker to ticketmaster format
def format_date(queried_date):
	start_date = queried_date.split("-")[0]
	end_date = queried_date.split("-")[1]
	start_date= str(dparser.parse(start_date))
	start_date = start_date.replace(" ", "T")+"Z"
	end_date = str(dparser.parse(end_date))
	end_date = end_date.replace(" ", "T")+"Z"
	return start_date, end_date

## ticketmaster format to readable format
def get_readable_date(date):
	date_obj = dt.datetime.strptime(date, '%Y-%m-%d')
	date = dt.datetime.strftime(date_obj,'%b %d, %Y')
	return date

## ticketmaster format to readable format
def get_readable_time(time):
	time_obj = dt.datetime.strptime(time, '%H:%M:%S')
	time = dt.datetime.strftime(time_obj,'%I:%M %p')
	return time

def format_output_message(queried_genre, queried_location, queried_date):
	if queried_genre == "music":
		queried_genre = ""
	else:
		queried_genre = queried_genre + " "
	if queried_location:
		queried_location = "in " + queried_location + " "
	else:
		queried_location = ""
	if queried_date:
		queried_date = "on " + queried_date
	else:
		queried_date = ""
	return "You're looking for " + queried_genre + "concerts " + queried_location + queried_date

