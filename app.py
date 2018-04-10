from app import app, socketio
import json
from flask import Flask, request, redirect, g, render_template
import requests
import base64
import urllib
import os

#### Code from https://github.com/drshrey/spotify-flask-auth-example
# Authentication Steps, paramaters, and responses are defined at https://developer.spotify.com/web-api/authorization-guide/
# Visit this url to see all the steps, parameters, and expected response. 


app = Flask(__name__)

template_dir = os.path.abspath('app/templates')
app = Flask(__name__, template_folder=template_dir)

#  Client Keys
CLIENT_ID = "3dc995c1706a437da9fb313f55498fee"
CLIENT_SECRET = "f81b54944e43434d9455065377455904"

# Spotify URLS
SPOTIFY_AUTH_URL = "https://accounts.spotify.com/authorize"
SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"
SPOTIFY_API_BASE_URL = "https://api.spotify.com"
API_VERSION = "v1"
SPOTIFY_API_URL = "{}/{}".format(SPOTIFY_API_BASE_URL, API_VERSION)


# Server-side Parameters
CLIENT_SIDE_URL = "http://127.0.0.1"
PORT = 5000
REDIRECT_URI = "{}:{}/callback".format(CLIENT_SIDE_URL, PORT)
#SCOPE = "playlist-modify-public playlist-modify-private"
SCOPE = "user-top-read"
STATE = ""
SHOW_DIALOG_bool = True
SHOW_DIALOG_str = str(SHOW_DIALOG_bool).lower()


auth_query_parameters = {
    "response_type": "code",
    "redirect_uri": REDIRECT_URI,
    "scope": SCOPE,
    # "state": STATE,
    # "show_dialog": SHOW_DIALOG_str,
    "client_id": CLIENT_ID
}

@app.route("/")
def index():
    # Auth Step 1: Authorization
    url_args = "&".join(["{}={}".format(key,urllib.quote(val)) for key,val in auth_query_parameters.iteritems()])
    auth_url = "{}/?{}".format(SPOTIFY_AUTH_URL, url_args)
    return redirect(auth_url)


@app.route("/callback")
def callback():
    # Auth Step 4: Requests refresh and access tokens
    auth_token = request.args['code']
    code_payload = {
        "grant_type": "authorization_code",
        "code": str(auth_token),
        "redirect_uri": REDIRECT_URI
    }
    base64encoded = base64.b64encode("{}:{}".format(CLIENT_ID, CLIENT_SECRET))
    headers = {"Authorization": "Basic {}".format(base64encoded)}
    post_request = requests.post(SPOTIFY_TOKEN_URL, data=code_payload, headers=headers)

    # Auth Step 5: Tokens are Returned to Application
    response_data = json.loads(post_request.text)
    print(response_data)
    access_token = response_data["access_token"]
    refresh_token = response_data["refresh_token"]
    token_type = response_data["token_type"]
    expires_in = response_data["expires_in"]

    # Auth Step 6: Use the access token to access Spotify API
    authorization_header = {"Authorization":"Bearer {}".format(access_token)}

    # # Get profile data
    # user_profile_api_endpoint = "{}/me".format(SPOTIFY_API_URL)
    # profile_response = requests.get(user_profile_api_endpoint, headers=authorization_header)
    # profile_data = json.loads(profile_response.text)

    # # Get user playlist data
    # playlist_api_endpoint = "{}/playlists".format(profile_data["href"])
    # playlists_response = requests.get(playlist_api_endpoint, headers=authorization_header)
    # playlist_data = json.loads(playlists_response.text)
    
    # # Get user top data
    top_artists_api_endpoint = "{}/me/top/artists?time_range=medium_term".format(SPOTIFY_API_URL)
    top_artists_response = requests.get(top_artists_api_endpoint, headers=authorization_header)
    top_artists_data = json.loads(top_artists_response.text)

    top_tracks_api_endpoint = "{}/me/top/tracks?time_range=medium_term".format(SPOTIFY_API_URL)
    top_tracks_response = requests.get(top_tracks_api_endpoint, headers=authorization_header)
    top_tracks_data = json.loads(top_tracks_response.text)

    # Combine profile and playlist data to display
    return render_template("index.html",artists_object=top_artists_data["items"], tracks_object=top_tracks_data["items"])
####

if __name__ == "__main__":
  print "Flask app running at http://0.0.0.0:5000"
  #app.debug = True
  socketio.run(app, host="0.0.0.0", port=5000)
