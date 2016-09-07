var timeline_height = 15;
var timeline_width  = 800;

function Timeline(parent) {
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

		// get smallest span and update zoom
		var new_min_span = Math.min.apply(null, logs.map(function(l){return l.end - l.beg;}));
		if (new_min_span < min_span){
			min_span = new_min_span;
			zoom.scaleExtent([1, (end - beg)/min_span]);
		}
		zoom.x(x);
		// separate logs into even and odd
		var even_data = [];
		var odd_data = [];
		var odd = true;

		for (var i in logs) {
			var log = logs[i];

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
			even_data.push(null);
			odd_data.push(null);
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

	this.onmousemove = function(x0){};

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

		// draw the x axis
		x_axis_g.call(x_axis);

		// draw the areas
		for (var i in timelines) {
			var tl = timelines[i];
			tl.odd_path.data([tl.odd_data]);
			tl.odd_path.attr('d', tl.area);
			tl.even_path.data([tl.even_data]);
			tl.even_path.attr('d', tl.area);
		}
	};

	var x = d3.time.scale()
		.range([margins.left, timeline_width+margins.left]);

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
		.attr('id', 'clip_rect_'+clip_rect_id).append('rect')
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
	var min_span = Infinity;
	// Return an object with methods to modify this thing
}

function render_timeline(parent, object, data) {
	var logs = object.data;
	var t = new Timeline(parent);
	t.prepare_timeline(logs);
	t.draw();

	/// make table
	var table = parent.append('table');

	var br = table.append('tr'), // begin row
		er = table.append('tr'), // end row
		vr = table.append('tr'); // value row

	br.append('th').text('begin');
	er.append('th').text('end');
	vr.append('th').text('value');

	var beg_td = br.append('td'),
		end_td = er.append('td'),
		val_td = vr.append('td');

	// create event listener that changes table data on mousemove
	t.onmousemove = function(x0){
		var log = get_rect_at(x0, object.data);

		beg_td.text(log ? format(log.beg) : null);
		end_td.text(log ? format(log.end) : null);
		val_td.text(log ? log.val : null);
	};
}

function render_timeline_diff(parent, object, data) {
	var diffs = object.data;
	var t = new Timeline(parent);

	classes = ['same', 'changed', 'added', 'removed'];
	for (var i in classes) {
		var cls = classes[i];
		var cls_diffs = diffs.filter(function(d){return difference(d) == cls;});
		if (cls_diffs.length === 0) continue;
		var tl = t.prepare_timeline(cls_diffs);
		console.log(tl);
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

	br.append('th').text('begin');
	er.append('th').text('end');
	or.append('th').text('old');
	nr.append('th').text('new');

	var beg_td = br.append('td'),
		end_td = er.append('td'),
		old_td = or.append('td'),
		new_td = nr.append('td');

	// create event listener that changes table data on mousemove
	t.onmousemove = function(x0){	
		var diff = get_rect_at(x0, object.data);

		beg_td.text(diff ? format(diff.beg) : null);
		end_td.text(diff ? format(diff.end) : null);
		old_td.text(diff ? diff.old : null);
		new_td.text(diff ? diff.new : null);
	};
}

function render_file_timeline(parent, object, data) {
	var ftimes = object.data;
	var t = new Timeline();
	t.prepare_timeline(ftimes);
	t.draw();

	/// make table
	var table = parent.append('table');

	var lr = table.append('tr'),
		dr = table.append('tr');

	lr.append('th').text('begin');
	lr.append('th').text('end');

	var beg_td = dr.append('td'),
		end_td = dr.append('td');

	// create event listener that changes table data on mousemove
	t.onmousemove = function(x0){
		var f = get_rect_at(x0, ftimes);

		beg_td.text(f ? format(f.beg) : null);
		end_td.text(f ? format(f.end) : null);
	};
}

function render_file_timeline_diff(parent, object, data) {
	var t = new Timeline(parent);
	var old_ftimes = [],
		new_ftimes = [];
		
	for (var i in object.old_ftimes) {
		old_ftimes.push({
			beg: object.old_ftimes[i][0],
			end: object.old_ftimes[i][1]
		});
	}
	for (var i in object.new_ftimes) {
		new_ftimes.push({
			beg: object.new_ftimes[i][0],
			end: object.new_ftimes[i][1]
		});
	}
	old_tl = t.prepare_timeline(old_ftimes);
	new_tl = t.prepare_timeline(new_ftimes);

	new_tl.area = d3.svg.area()
		.defined(new_tl.area.defined())
		.x(new_tl.area.x())
		.y1(timeline_height)
		.y0(timeline_height*2);

	/// create the elements so we can adjust then
	t.draw();
	/// adjust timeline dimensions
	svg = parent.select('svg')
		.attr('height', timeline_height*2+margins.bottom);

	svg.select('g.x.axis')
		.attr('transform', 'translate(0,'+timeline_height*2+')');

	svg.select('rect')
		.attr('height', timeline_height*2);

	svg.select('rect.listener')
		.attr('height', timeline_height*2);

	/// make table
	var table = parent.append('table');

	var lr = table.append('tr'),
		or = table.append('tr'),
		nr = table.append('tr');

	lr.append('td');
	lr.append('th').text('begin');
	lr.append('th').text('end');

	or.append('th').text('old');
	nr.append('th').text('new');

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
}