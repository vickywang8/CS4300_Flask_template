from . import *  
from app.irsystem.models.helpers import *
from app.irsystem.models.helpers import NumpyEncoder as NumpyEncoder

project_name = "ConcertMaster"
net_id = "Minzhi Wang: mw787, Emily Sun: eys27, Priyanka Rathnam: pcr43, Lillyan Pan: ldp54, Rachel Kwak: sk2472"

@irsystem.route('/', methods=['GET'])
def search():
	query = request.args.get('search')
	if not query:
		data = []
		output_message = ''
	else:
		output_message = "Your search: " + query
		data = range(5)
	return render_template('search.html', name=project_name, netid=net_id, output_message=output_message, data=data)



