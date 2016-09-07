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