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

areas_dict = {
        'Birmingham & More': "1", 'Charlotte' : "2", 'Chicagoland & Northern IL': "3", 'Cincinnati & Dayton': "4",
        'Dallas - Fort Worth & More': "5", 'Denver & More': "6", 'Detroit, Toledo & More': "7", 
        'El Paso & New Mexico': "8", 'Grand Rapids & More': "9", 'Greater Atlanta Area': "10", 
        'Greater Boston Area': "11", 'Cleveland, Youngstown & More': "12", 'Greater Columbus Area': "13",
        'Greater Las Vegas Area': "14", 'Greater Miami Area': "15", 'Minneapolis/St. Paul & More': "16", 
        'Greater Orlando Area': "17", 'Greater Philadelphia Area': "18", 'Greater Pittsburgh Area': "19", 
        'Greater San Diego Area': "20", 'Greater Tampa Area': "21", 'Houston & More': "22", 'Indianapolis & More': "23",
        'Iowa': "24", 'Jacksonville & More': "25", 'Kansas City & More': "26", 'Greater Los Angeles Area': "27",
        'Louisville & Lexington': "28", 'Memphis, Little Rock & More': "29", 'Milwaukee & WI': "30", 
        'Nashville, Knoxville & More': "31", 'New England': "33", 'New Orleans & More': "34", 'New York/Tri-State Area': "35",
        'Phoenix & Tuscon': "36", 'Portland & More': "37", 'Raleigh & Durham': "38", 'Saint-Louis & More': "39",
        'San Antonio & Austin': "40", 'N.California/N.Nevada': "41", 'Greater Seattle Area': "42", 
        'North & South Dakota': "43", 'Upstate New York': "44", 'Utah & Montana': "45", 'Virginia': "46", 
        'Washington, DC and Maryland': "47", 'West Virginia': "48", 'Hawaii': "49", "Alaska": "50",
        'Nebraska': "52", 'Springfield': "53", 'Central Illinois': "54", 'Northern New Jersey': "55",
        'South Carolina': "121", 'South Texas': "122", 'Beaumont': "123", 'Connecticut': "124", 'Oklahoma': "125"
      }

@irsystem.route('/', methods=['GET'])
def search():
	queried_artist = request.args.get('artist')
	queried_genre = request.args.get('genre')
	queried_area = request.args.get('area')
	queried_city = request.args.get('city')
	queried_date = request.args.get('date')
	data = None
	output_message = ""
	if not (queried_artist or queried_genre or queried_area or queried_city or queried_date):
		data = []
	else:
		if not queried_genre:
			queried_genre = "music"
		if queried_date:
			start_date, end_date = format_date(queried_date)
		search_endpoint = "{}events.json?classificationName={}&city={}&marketId={}&countryCode=US&startDateTime={}&endDateTime={}&keyword={}&includeSpellcheck=yes{}".format(TICKETMASTER_API_URL, queried_genre, queried_city, areas_dict[queried_area], start_date, end_date, queried_artist, TICKETMASTER_API_KEY)
		try:
			search_response = requests.get(search_endpoint)
			search_data = json.loads(search_response.text)
			if "spellcheck" in search_data.keys():
				print(search_data)
				spellchecked_artist = search_data["spellcheck"]["suggestions"][0]["suggestion"]
				search_endpoint = "{}events.json?classificationName={}&city={}&marketId={}&countryCode=US&startDateTime={}&endDateTime={}&keyword={}&includeSpellcheck=yes{}".format(TICKETMASTER_API_URL, queried_genre, queried_city, areas_dict[queried_area], start_date, end_date, spellchecked_artist, TICKETMASTER_API_KEY)
				search_response = requests.get(search_endpoint)
				search_data = json.loads(search_response.text)	
			data = get_concert_data(search_data)	
		except Exception as e:
			date = []
			output_message = "Your search returned no results. Modify your search and try again!"
			print(e)
	return render_template('search.html', name=project_name, netid=net_id, output_message=output_message, data=data, queried_genre= queried_genre, queried_date=queried_date, queried_city=queried_city, queried_area=queried_area, queried_artist=queried_artist)

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
		venue = event["_embedded"]["venues"][0]["name"]
		city = event["_embedded"]["venues"][0]["city"]["name"]
		concert_data = {"name": name, "date": date, "time": time, "link": link, "venue": venue, "city": city}
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

def format_output_message(spellchecked_artist, queried_genre, queried_city, queried_area, queried_date):
	if spellchecked_artist:
		spellchecked_artist = spellchecked_artist + " "
	else:
		spellchecked_artist = ""
	if queried_genre == "music":
		queried_genre = ""
	else:
		queried_genre = queried_genre + " "
	if queried_city or queried_area:
		location_present = "in "
		if queried_city:
			queried_city = queried_city + " "
		else:
			queried_city = ""
		if queried_area:
			queried_area = queried_area + " "
		else:
			queried_area = ""
	else:
		location_present = ""
	if queried_date:
		queried_date = "on " + queried_date
	else:
		queried_date = ""
	return "You're looking for " + spellchecked_artist + queried_genre + "concerts " + location_present + queried_city + queried_area + queried_date

