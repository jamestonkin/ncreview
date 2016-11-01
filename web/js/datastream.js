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
	var details = parent.append('div');
	details.append('h1').text('Summary').style('color', '#ff6600');
	
	details = details.append('div')
		//.style('padding-left', '1.25em')
		.style('border-top', '1px solid black');

	if(object['bad_data'].length != 0) {
		var par = details.append('p').text('Bad Data. The sums of each old/new column in each chart in Variables.');
		var table = details.append('table').attr('id', 'bad_data');

		for(let key of ['nmiss', 'nanns', 'infs', 'fills']) {
			var tr = table.append('tr');
			tr.append('th').text(key).style('color', object['bad_data'][key] > 0 ? '#22b' : null).style('text-align', 'right');
			tr.append('td').text(object['bad_data'][key]);
		}
	}

	if(object['different_times'].length != 0){
		var par = details.append('p').text('Changes in Dimensions-time. The hard-to-see blue lines in the timeline.');

		var table = details.append('table');
		var tr = table.append('tr');
		tr.append('th').text('Date').style('text-align', 'left');
		tr.append('th').text('Old').style('text-align', 'right');
		tr.append('th').text('New').style('text-align', 'right');
		tr.append('th').text('Diff').style('text-align', 'right');

		for (var i in object['different_times']) {
			var arr = object['different_times'][i];
			var date = arr[0];
			var old = arr[1];
			var _new = arr[2];
			var diff = arr[3];

			var tr2 = table.append('tr');
			tr2.append('td').text(String(epoch2utc(date)).substring(4, 15));
			tr2.append('td').text(old).style('text-align', 'right');
			tr2.append('td').text(_new).style('text-align', 'right');
			tr2.append('td').text(diff).style('color', diff > 0 ? '#00642e' : 'b30006').style('text-align', 'right');
		}
	}
}