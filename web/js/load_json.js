data_id = null;
datastream = null;
/**
 * Get data id from the url and load it
 */
function load_from_url() {
	id = window.location.href.split('?')[1];

	if (id) {
		data_id = id;
		load_json(id);
	}
}

/**
 * Loads the specified json file and renders it using render_object
 */
function load_json(id) {
	var status_indicator = d3.select('#status_indicator');
	status_indicator.text('Loading Data...');
	status_indicator.append('br');
	status_indicator.append('progress');
	d3.json('data.php?id='+id, function(error, data) {
		if (error) {
			console.error(error);
			status_indicator.text('Load Failed. See console for details.');
		}
		else {
			status_indicator.remove();
			datastream = data;
			render_object(d3.select('#main'), data, data);
		}
	});
}

function load_csv(parent, id, num, callback) {
	parent.append('progress');
	d3.csv('data.php?id='+id+'&num='+num, function(error, data) {
		parent.select('progress').remove();
		if (error) {
			console.error(error);
			parent.append('p')
				.text('Load failed. See console for details')
				.style('color', 'red');
			return;
		}
		callback(data);
	});
}		