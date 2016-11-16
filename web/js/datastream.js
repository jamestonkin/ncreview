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

	if (Object.keys(object['bad_data']).length === 0 && Object.keys(object['data']).length === 0) {
		return;
	}
	
	var main_div = parent.append('div');
	main_div.append('h1').text('Summary').style('color', '#ff6600');
	
	div = main_div.append('div')
		.style('border-top', '1px solid black');

	if(Object.keys(object['bad_data']).length != 0) {

		var par = div.append('p').text('Old Bad Data. The sums of each old column of each chart in Variables.');
		var table = div.append('table').attr('id', 'bad_data');

		for(let key of ['nmiss', 'nanns', 'infs', 'fills']) {
			var tr = table.append('tr');
			tr.append('th').text(key).style('color', object['bad_data'][key] > 0 ? '#22b' : null).style('text-align', 'right');
			tr.append('td').text(object['bad_data'][key]);
		}
	}

	if(Object.keys(object['bad_data2']).length != 0) {
		var par = div.append('p').text('New Bad Data. The sums of each new column of each chart in Variables.');
		var table = div.append('table').attr('id', 'bad_data');

		for(let key of ['nmiss', 'nanns', 'infs', 'fills']) {
			var tr = table.append('tr');
			tr.append('th').text(key).style('color', object['bad_data2'][key] > 0 ? '#22b' : null).style('text-align', 'right');
			tr.append('td').text(object['bad_data2'][key]);
		}
	}

	

	if(Object.keys(object['data']).length != 0) {

		LENGTH = 0;
		for(var x in object['data']['Header']) {
			LENGTH = x;
		}

		var sorted = [];
		for(var key in object['data']) {
    		sorted[sorted.length] = key;
		}
		sorted.sort();

		var par = div.append('p').text('A more-detailed analysis of misses, NaNs, INFs, and fills.');
		var table = div.append('table').attr('id', 'data');

		if(LENGTH > 5) {
			var tr = table.append('tr');
			tr.append('td').text('');
			tr.append('td').text('OLD DATA').attr('colspan', '4').style('text-align', 'center');
			tr.append('td').text('');
			tr.append('td').text('NEW DATA').attr('colspan', '4').style('text-align', 'center');
		}

		tr = table.append('tr');

		for(var value of object['data']['Header']) {
			tr.append('th').text(value).style('text-align', 'left');
		}

		for(var key of sorted) {
			if(key === 'Total' || key === 'Header') {
				continue;
			}

			tr = table.append('tr');
			tr.append('td').text(key);
			var i = 0;
			for(var value of object['data'][key]) {
				var cls = 'same';

				if (i >=1 && i <= 4 && LENGTH > 5) {
					if(value === object['data'][key][i + 5]) {
						cls = 'same';
					}
					else if(value > object['data'][key][i + 5]) {
						cls = 'added';	// green
					}
					else if (value < object['data'][key][i + 5]){
						cls = 'removed';	// red
					}	
				}
				else if (i >= 6 && i <= 9 && LENGTH > 5) {
					if(value === object['data'][key][i - 5]) {
						cls = 'same';
					}
					else if(value > object['data'][key][i - 5]) {
						cls = 'added';
					}
					else if (value < object['data'][key][i - 5]){
						cls = 'removed';
					}	
				}

				tr.append('td').text(value).style('text-align','right').attr('class', cls);
				i++;
			}
		}

		var i = 0;
		tr = table.append('tr');
		tr.append('td').text('Total').style('text-align', 'left').style('font-weight', 'bold');
		for(var value of object['data']['Total']) {
			var cls = 'same';
			if (i >=0 && i <= 3 && LENGTH > 5) {
				if(value === object['data']['Total'][i + 5]) {
					cls = 'same';
				}
				else if(value > object['data']['Total'][i + 5]) {
					cls = 'added';
				}
				else if (value < object['data']['Total'][i + 5]){
					cls = 'removed';
				}	
			}
			else if (i >= 5 && i <= 8 && LENGTH > 5) {
				if(value === object['data']['Total'][i - 5]) {
					cls = 'same';
				}
				else if(value > object['data']['Total'][i - 5]) {
					cls = 'added';
				}
				else if (value < object['data']['Total'][i - 5]){
					cls = 'removed';
				}	
			}

			tr.append('td').text(value).style('text-align', 'right').style('font-weight', 'bold').attr('class', cls);
			i++;
		}
		
	}

	if(Object.keys(object['different_times']).length != 0){
		var par = div.append('p').text('Changes in Dimensions-time. The hard-to-see blue lines in the timeline.');

		var table = div.append('table');
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
			tr2.append('td').text(diff).style('color', diff > 0 ? '00642e' : 'b30006').style('text-align', 'right');
		}
	}
}