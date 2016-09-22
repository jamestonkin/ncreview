var timeline_height = 15;
var timeline_width  = 800;
//---------------------------------------------------------------------------------------------------------------------
/**
 * Generic Timeline renderer.
 * This class sets up many of the components nessecary to create a timeline,
 * and returns an object with some methods for editing the timeline
 */
function Timeline(parent) {
	/**
	 * Set up objects to draw an array of logs in the timeline.
	 * Expects each log to have a beginning and end time.
	 */
	this.prepare_timeline = function(logs){
		for (var i in logs) {
			var log = logs[i];

			log.beg = epoch2utc(log.beg);
			log.end = epoch2utc(log.end);
		}
		// get earliest and latest dates and update x
		if (beg === undefined || logs[0].beg < beg            ) beg = logs[0].beg;
		if (end === undefined || logs[logs.length-1].end > end) end = logs[logs.length-1].end;

		x.domain([beg, end]);

		zoom.x(x);
		// separate logs into even and odd for coloring
		var even_data = [];
		var odd_data = [];
		var odd = true;
	
		for (var i in logs) {
			var log = logs[i];
			all_logs.push(log);

			if (log !== null) {
				if (odd) {
					odd_data.push(log.beg);
					odd_data.push(log.end);
				}
				else {
					even_data.push(log.beg);
					even_data.push(log.end);
				}
				odd = !odd;
			}
			if (even_data.length && even_data[even_data.length-1]) even_data.push(null);
			if (odd_data.length  && odd_data [odd_data.length-1])  odd_data.push(null);
		}
		// create new svg paths
		var even_path = tg.append('path')
			.attr('class', 'same diff even')
			.attr('clip-path', 'url(#clip_rect_'+clip_rect_id+')');

		var odd_path = tg.append('path')
			.attr('class', 'same diff odd')
			.attr('clip-path', 'url(#clip_rect_'+clip_rect_id+')');

		// store the results in this data struct and return it for customization
		var timeline_data = {
			'odd_data' : odd_data,
			'even_data': even_data,
			'odd_path' : odd_path,
			'even_path': even_path,
			'area'     : area
		};

		timelines.push(timeline_data);
		return timeline_data;
	};

	/**
	 * Specific timelines assign a function to this attribute which is called when the timeline is moused over.
	 */
	this.onmousemove = function(x0){};

	/**
	 * Create the timeline svg and elements
	 */
	this.draw = function(){
		// make sure transform is within bounds
		var t = zoom.translate(),
			tx = t[0],
			ty = t[1];

		var left_overshoot = x(beg) - margins.left;

		if (left_overshoot > 0)
			tx -= x(beg) - margins.left;

		var right_overshoot = x(end) - (timeline_width+margins.left);
		if (right_overshoot < 0) {
			tx -= x(end) - (timeline_width+margins.left);
		}
		zoom.translate([tx, ty]);

		// limit the zoom to the smallest span in the window
		var l_beg = x.invert(margins.left), // l for local begin and end
			l_end = x.invert(margins.left+timeline_width);

		// get all logs currently in the window
		function in_window(l){
			return l_beg <= l.beg && l.beg <= l_end && l_beg <= l.end && l.end <= l_end;
		}
		var l_logs = all_logs.filter(in_window);

		// get the minimum span of any of these logs, or the window size if there are no logs.
		var min_span = l_end-l_beg;
		if(l_logs.length) {
			min_span = Math.abs(Math.min(min_span, d3.min(l_logs.map(function(l){return l.end - l.beg;}))));
		}
		// limit the scale of the timeline to the smallest thing in view
		zoom.scaleExtent([1, (end-beg)/min_span]);

		// draw the x axis
		x_axis_g.call(x_axis);

		// draw the areas
		for (var j in timelines) {
			var tl = timelines[j],
				odd_data  = tl.odd_data,
				even_data = tl.even_data;

			// get the data within the current window
			var visible_odd_data  = [];
			var visible_even_data = [];

			i = 0;
			while(i < odd_data.length && odd_data[i] < l_beg) i++;

			if (i > 0) visible_odd_data.push(odd_data[i-1]);

			while(i < odd_data.length && odd_data[i] < l_end) {
				visible_odd_data.push(odd_data[i]);
				i++;
			}
			if (i < odd_data.length) visible_odd_data.push(odd_data[i]);

			i = 0;
			while(i < even_data.length && even_data[i] < l_beg) i++;

			if (i > 0) visible_even_data.push(even_data[i-1]);

			while(i < even_data.length && even_data[i] < l_end) {
				visible_even_data.push(even_data[i]);
				i++;
			}

			if (i < even_data.length) visible_even_data.push(even_data[i]);
			
			tl.odd_path.data([visible_odd_data]);
			tl.odd_path.attr('d', tl.area);
			tl.even_path.data([visible_even_data]);
			tl.even_path.attr('d', tl.area);
		}
	};

	var x = d3.time.scale()
		.rangeRound([margins.left, timeline_width+margins.left]);

	var area = d3.svg.area()
		.defined(function(d) {return d !== null;})
		.x(function(d){return x(d);})
		.y1(0)
		.y0(timeline_height);

	var x_axis = d3.svg.axis()
		.scale(x)
		.orient('bottom');

	var zoom = d3.behavior.zoom()
		.on('zoom', this.draw);

	var timelines = [];

	// main svg
	var svg = parent.append('svg')
		.attr('height', timeline_height+margins.bottom)
		.attr('width', timeline_width+margins.left+margins.right);

	// clipping rect
	var clip_rect_id = ++unique_id;
	var clip_rect = svg.append('clipPath')
		.attr('id', 'clip_rect_'+clip_rect_id)
		.attr('class', 'clip_path').append('rect')
			.attr('width', timeline_width)
			.attr('height', timeline_height)
			.attr('x', margins.left)
			.attr('y', 0)
			.style('fill', 'none');

	// timeline g
	var tg = svg.append('g')
		.attr('class', 'timeline_g');

	// x axis
	var x_axis_g = svg.append('g')
		.attr('class', 'x axis')
		.attr('transform', 'translate(0,'+timeline_height+')');

	// event listener rect
	var listen_rect = svg.append('rect')
		.attr('class', 'listener')
		.attr('x', margins.left)
		.attr('y', 0)
		.attr('width', timeline_width)
		.attr('height', timeline_height)
		.call(zoom);

	var me = this;
	listen_rect.on('mousemove', function(){
		var x0 = x.invert(d3.mouse(this)[0]);
		me.onmousemove(x0);
	});

	var beg, end;
	var all_logs = [];
	// Return an object with methods to modify this thing
}
//---------------------------------------------------------------------------------------------------------------------
/**
 * Render a timeline for a list of values, not a comparison
 */
function render_timeline(parent, object) {
	load_csv(parent, data_id, object.data, function(logs){
		var t = new Timeline(parent);
		t.prepare_timeline(logs);
		t.draw();

		/// make table
		var table = parent.append('table');

		var br = table.append('tr'), // begin row
			er = table.append('tr'), // end row
			vr = table.append('tr'); // value row

		br.append('th').text('Begin');
		er.append('th').text('End');
		vr.append('th').text('Value');

		var beg_td = br.append('td'),
			end_td = er.append('td'),
			val_td = vr.append('td');

		// create event listener that changes table data on mousemove
		t.onmousemove = function(x0){
			var log = get_rect_at(x0, logs);

			beg_td.text(log ? format(log.beg) : null);
			end_td.text(log ? format(log.end) : null);
			val_td.html(log && log.val !== "" ? log.val : "<i>N/A</i>");
		};
	});
}
//---------------------------------------------------------------------------------------------------------------------
/**
 * Render a timeline for a comparison of old and new values
 */
function render_timeline_diff(parent, object) {
	load_csv(parent, data_id, object.data, function(diffs){
		var t = new Timeline(parent);

		classes = ['same', 'changed', 'added', 'removed'];
		for (var i in classes) {
			var cls = classes[i];
			var cls_diffs = diffs.filter(function(d){return difference(d) == cls;});
			if (cls_diffs.length === 0) continue;
			var tl = t.prepare_timeline(cls_diffs);
			tl.odd_path.attr('class', cls+' diff odd');
			tl.even_path.attr('class', cls+' diff even');
		}
		t.draw();

		/// make table
		var table = parent.append('table');

		var br = table.append('tr'), // begin row
			er = table.append('tr'), // end row
			or = table.append('tr'), // old row
			nr = table.append('tr'); // new row

		br.append('th').text('Begin');
		er.append('th').text('End');
		or.append('th').text('Old');
		nr.append('th').text('New');

		var beg_td = br.append('td'),
			end_td = er.append('td'),
			old_td = or.append('td'),
			new_td = nr.append('td');

		// create event listener that changes table data on mousemove
		t.onmousemove = function(x0){
			var diff = get_rect_at(x0, diffs);
			beg_td.text(diff ? format(diff.beg) : null);
			end_td.text(diff ? format(diff.end) : null);
			old_td.html(diff && diff.old !== "" ? diff.old : "<i>N/A</i>");
			new_td.html(diff && diff.new !== "" ? diff.new : "<i>N/A</i>");
		};
	});
}
//---------------------------------------------------------------------------------------------------------------------
/*
 * Render a file timeline for a single datastream
 */
function render_file_timeline(parent, object) {
	load_csv(parent, data_id, object.data, function(ftimes) {
		var t = new Timeline(parent);
		t.prepare_timeline(ftimes);
		t.draw();

		/// make table
		var table = parent.append('table');

		var lr = table.append('tr'),
			dr = table.append('tr');

		lr.append('th').text('Begin');
		lr.append('th').text('End');

		var beg_td = dr.append('td'),
			end_td = dr.append('td');

		// create event listener that changes table data on mousemove
		t.onmousemove = function(x0){
			var f = get_rect_at(x0, ftimes);

			beg_td.text(f ? format(f.beg) : null);
			end_td.text(f ? format(f.end) : null);



		};
	});
}
//---------------------------------------------------------------------------------------------------------------------
/**
 * Render a file timeline comparing two datastream's files, side-by-side.
 */
function render_file_timeline_diff(parent, object) {
	load_csv(parent, data_id, object.old_data, function(old_ftimes){
		load_csv(parent, data_id, object.new_data, function(new_ftimes){
			var t = new Timeline(parent);
			old_tl = t.prepare_timeline(old_ftimes);
			new_tl = t.prepare_timeline(new_ftimes);

			new_tl.area = d3.svg.area()
				.defined(new_tl.area.defined())
				.x(new_tl.area.x())
				.y1(timeline_height)
				.y0(timeline_height*2);

			// create the elements so we can adjust them
			t.draw();

			/// adjust timeline
			svg = parent.select('svg')
				.attr('height', timeline_height*2+margins.bottom);

			svg.select('g.x.axis')
				.attr('transform', 'translate(0,'+timeline_height*2+')');

			svg.select('.clip_path').select('rect')
				.attr('height', timeline_height*2);

			svg.select('rect.listener')
				.attr('height', timeline_height*2);

			// add old and new labels
			svg.append('text')
				.text('Old:')
				.attr('y', timeline_height);

			svg.append('text')
				.text('New:')
				.attr('y', timeline_height*2);

			/// make table
			var table = parent.append('table');

			var lr = table.append('tr'),
				or = table.append('tr'),
				nr = table.append('tr');

			lr.append('td');
			lr.append('th').text('Begin');
			lr.append('th').text('End');

			or.append('th').text('Old');
			nr.append('th').text('New');

			var old_beg_td = or.append('td'),
				old_end_td = or.append('td'),
				new_beg_td = nr.append('td'),
				new_end_td = nr.append('td');

			// create event listener that changes table data on mousemove
			t.onmousemove = function(x0){
				var oldf = get_rect_at(x0, old_ftimes);
				var newf = get_rect_at(x0, new_ftimes);

				old_beg_td.text(oldf ? format(oldf.beg) : null);
				old_end_td.text(oldf ? format(oldf.end) : null);
				new_beg_td.text(newf ? format(newf.beg) : null);
				new_end_td.text(newf ? format(newf.end) : null);
			};
		});
	});
}