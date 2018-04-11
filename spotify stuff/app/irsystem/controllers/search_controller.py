from . import *  
from app.irsystem.models.helpers import *
from app.irsystem.models.helpers import NumpyEncoder as NumpyEncoder

project_name = "ConcertMaster"
net_id = "Minzhi Wang: mw787, Emily Sun: eys27, Priyanka Rathnam: pcr43, Lillyan Pan: ldp54, Rachel Kwak: sk2472"

@irsystem.route('/callback', methods=['GET'])
def search():
	query = request.args.get('search')
	print(query)
	if not query:
		data = []
		output_message = ''
	else:
		output_message = "Your search: " + query
		data = range(5)
	return render_template('index.html', name=project_name, netid=net_id, output_message=output_message, data=data)



