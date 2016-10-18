/**
 * Top-level render function for a single datastream
 */
function render_datastream(parent, object) {
	// render header
	parent.append('h1')
		.text(object.path);

	// render contents
	for (var i in object.contents)
        render_object(parent, object.contents[i]);
}

/**
 * Top-level render function for a comparison of datastreams
 */
function render_datastream_diff(parent, object) {
	// render header
	parent.append('h1')
		.html(object.old_path +
			 ' <i>(old)<br>Vs.</i><br>' +
			 object.new_path+' <i>(new)</i>');

	// Expand/collapse sections button
	parent.append('button')
		.text('Expand Changed Sections')
		.on('click', function() {
			d3.select('#main').selectAll('summary.changed')
				.each(function(){
					d3.select(this.parentNode).attr('open', '');
					try {
						d3.select(this).on('click')();
					}
					catch(err) {}
				});
		});
	parent.append('br');
	
	// render contents
	for (var i in object.contents)
        render_object(parent, object.contents[i]);
}


function render_summary(parent, object) {
	// render the summary details menu
		if(!(object['bad_data']['nanns'] == -1 && object['bad_data']['infs'] == -1 && object['bad_data']['fills'] == -1)){
			var details = parent.append('div');
			/*
			var summary = details.append('summary')
				.append('b')
				.text('Summary');
			*/

			details = details.append('div')
				.style('padding-left', '1.25em');

			if(object['bad_data'] != []) {
				var par = details.append('p').text('Bad Data');
				var table = details.append('table').attr('id', 'bad_data');

			for(let key of ['nanns', 'infs', 'fills']) {
				var tr = table.append('tr');
				tr.append('th').text(key).style('color', object['bad_data'][key] != 0 ? '#22b' : null);
				tr.append('td').text(object['bad_data'][key]);
			}
		}
	}

	if(object['different_times'].length != 0){
		var table = details.append('table');
		var caption = table.append('caption')
			.text('Changes in Dimensions-time');
		// note that we can read data from object['ranodm_text']
		var tr = table.append('tr');
		tr.append('th').text('Date');
		tr.append('th').text('Old');
		tr.append('th').text('New');
		tr.append('th').text('Diff');

		for (var i in object['different_times']) {
			var arr = object['different_times'][i];
			var date = arr[0];
			var old = arr[1];
			var _new = arr[2];
			var diff = arr[3];

			var tr2 = table.append('tr');
			tr2.append('td').text(String(epoch2utc(date)).substring(4, 15));
			tr2.append('td').text(old);
			tr2.append('td').text(_new);
			tr2.append('td').text(diff);
		}
	}
}