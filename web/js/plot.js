plot_svg_width = 1000;
plot_svg_height = 400;
plot_width = plot_svg_width - margins.left - margins.right;
plot_height = plot_svg_height - margins.top - margins.bottom;

unique_id = 0;

function convert_dates(logs){
	for (var i in logs){
		for (var j in logs[i]) {
			logs[i][j] = logs[i][j] !== '' ? +logs[i][j] : null;
		}
		logs[i].beg = epoch2utc(logs[i].beg);
		logs[i].end = epoch2utc(logs[i].end);
	}
}

function dq_inspector_plot(parent, dir, var_name, beg, end, version) {
	var php_query = 'dq_inspector.php?dir='+dir+'&var='+var_name+'&beg='+beg+'&end='+end;
	if (version) php_query += '&version='+version;
	parent.append('progress');
	return d3.json(php_query, function(error, data) {
		parent.select('progress').remove();
		if (error) {
			console.error(error);
			parent.append('div').html(error.response).style('color', 'red');
		}
		else {
			if (version) parent.append('h3').text(version);
			parent.append('p').text(data.cmd);
			parent.append('img')
				.attr('src', 'dq_plot.php?plot='+data.plot)
				.style('height', 'auto')
				.style('width', '800px');
		}
	});
}

// TODO: plot multiple values
// TODO: user-specified y-axis scaling
//---------------------------------------------------------------------------------------------------------------------

function plot(parent, columns, tooltips, var_name, ds_path, old_ds_path, new_ds_path) {
	var obj = {};
	var beg = Infinity,
		end = -Infinity;

	var key = columns.indexOf('mean') > -1 ? 'mean' : columns[0];

	var datas = [];

	var x = d3.time.scale()
		.range([margins.left, plot_width+margins.left]);

	var y = d3.scale.linear()
		.range([plot_height+margins.top, margins.top]);

	// create axes
	var xAxis = d3.svg.axis()
			.scale(x)
			.orient('bottom');

	var yAxis = d3.svg.axis()
			.scale(y)
			.orient('left');

	// create line
	var line = d3.svg.line()
		.defined(function(l) {return l[key] !== null;});

	// create area
	var area = d3.svg.area()
		.defined(function(t) {return t !== null;})
		.y0(plot_height+margins.top)
		.y1(margins.top);

	//create zoom 
	var zoom = d3.behavior.zoom();

	/// Create svg and html elements for table ///

	// main svg
	var svg = parent.append('svg')
		.attr('width', plot_svg_width)
		.attr('height', plot_svg_height)
		.attr('class', 'plot');

	// x axis container
	var xAxisG = svg.append('g')
		.attr('class', 'x axis')
		.attr('transform',
			'translate(0,'+(plot_svg_height-margins.bottom)+')'
			);
	
	// y axis container
	var yAxisG = svg.append('g')
		.attr('class', 'y axis')
		.attr('transform',
			'translate('+margins.left+',0)');

	// clipping rect
	var clip_id = ++unique_id;
	var clip_rect = svg.append('clipPath')
		.attr('id', 'clip'+clip_id).append('rect')
			.attr('width', plot_width)
			.attr('height', plot_height)
			.attr('x', margins.left)
			.attr('y', margins.top)
			.style('fill', 'none');

	var path_g = svg.append('g');
	
	var paths = [];
	var bg_areas = {};
	
	// text displaying the selected date
	var mouse_date = svg.append('text')
		.attr('y', margins.top+5)
		.attr('class', 'mouse date');

	// vertical line at the selected date
	var mouse_line = svg.append('line')
		.attr('y1', margins.top)
		.attr('y2', plot_height+margins.top)
		.attr('class', 'date line');

	// pane which holds event listeners for plot
	var listen_pane = svg.append('rect')
		.attr('class', 'listener')
		.attr('width', plot_width)
		.attr('height', plot_height)
		.attr('x', margins.left)
		.attr('y', margins.top)
		.call(zoom);

	/// create the table
	var table = parent.append('table');
	var hr = table.append('tr');
	hr.append('td');
	hr.selectAll('th').data(columns).enter().append('th')
		.text(function(d){return d})
		.attr('title', function(d, i){return tooltips[i];})
		.style('border', function(d){
			if (d == key) {
				return '1px solid black';
			}
			return 'none';
		})
		.on('click', function(d){
			hr.selectAll('th').style('border', 'none');
			d3.select(this).style('border', '1px solid black');
			key = d;
			obj.draw();
		})
		.on('mouseover', function(d, i){
			// select this column of tds and expand them
			var col = table.selectAll('td.r'+i);
			col.selectAll('.short').attr('hidden', '');
			col.selectAll('.long').attr('hidden', null);
		})
		.on('mouseout', function(d, i){
			var col = table.selectAll('td.r'+i);
			col.selectAll('.short').attr('hidden', null);
			col.selectAll('.long').attr('hidden', '');
		});

	var rows = [];

	listen_pane.on('mousemove', function(){
		var x0 = x.invert(d3.mouse(this)[0]);
		var rect = get_rect_at(x0, rows[0].logs);
		if (rect === null) return;
		var log_x = mean_time(rect);
		mouse_date
			.text(format(log_x))
			.attr('x', x(log_x))
			.attr('hidden', null);

		mouse_line
			.attr('x1', x(log_x))
			.attr('x2', x(log_x));

		for (var i in rows) {
			row = rows[i];
			var log = get_rect_at(x0, row.logs);
			log_vals = log ? 
				columns.map(function(key){return log[key];}) : 
				columns.map(function(k){return null;});
			row.tr.selectAll('td').data(log_vals).html(function(d){
				if (d === null || d === undefined) {
					return '<i>N/A</i>';
				}
				var html = '<span class="short">'+Math.round(d*1000)/1000+'</span>';
				html += '<span class="long" hidden>'+d+'</span>';
				return html;
			});
		}
		// diff plot only:
		if (rows.length == 2) {
			hr.selectAll('th')
				.attr('class', function(d, i){
					var o = table.select('tr:nth-child(2)').select('td.r'+i).select('.long');
					var n = table.select('tr:nth-child(3)').select('td.r'+i).select('.long');
					o = o.empty() ? null : o.text();
					n = n.empty() ? null : n.text();

					return difference(o, n);
				});
		}
	});
	
	// create dq inspector plots button
	var dq_inspector_requests = [];
	if (var_name) {
		var dq_button = parent.append('button').text('dq inspector plots');
		var dq_div = parent.append('div');
		dq_button.on('click', function(){
				// cancel any existing requests
				for(var i in dq_inspector_requests) {
					dq_inspector_requests[i].abort();
				}
				dq_inspector_requests = [];
				// clear any existing plots
				dq_div.html('');
				// get plot start and end dates
				var date_range = x.domain();
				var beg = date_range[0];
				var end = date_range[1];
				beg = format(beg);
				end = format(end);

				// create the plots
				if (ds_path) {
					dq_inspector_requests.push(
						dq_inspector_plot(
							dq_div,
							ds_path,
							var_name,
							beg, end)
						);
				}
				else {
					dq_inspector_requests.concat([
						dq_inspector_plot(
						dq_div,
						old_ds_path,
						var_name,
						beg, end, 'old'),
					dq_inspector_plot(
						dq_div,
						new_ds_path,
						var_name,
						beg, end, 'new')

					]);
				}
			});
	}

	/// create handle object

	obj.add_line = function(logs, cls){
		if (cls === undefined) cls = '';
		beg = Math.min(beg, logs[0].beg);
		end = Math.max(end, logs[logs.length-1].end);
		x.domain([beg, end]);
		line.x(function(l){return x(mean_time(l));});
		zoom.x(x);
		datas.push(logs);
		paths.push(path_g.append('path')
			.attr('class', cls+' line')
			.attr('clip-path', 'url(#clip'+clip_id+')'));
	};

	obj.add_background = function(){
		classes = ['same', 'changed', 'added', 'removed'];
		for (var i in classes) {
			var cls = classes[i];
			bg_areas[cls] = path_g.append('path')
				.attr('class', cls+' area')
				.attr('clip-path', 'url(#clip'+clip_id+')');
		}

	};

	obj.add_row = function(label, logs){
		var row = table.append('tr');
		row.append('th').text(label);
		for (var i in columns) {
			row.append('td').attr('class', 'r'+i);
		}
		rows.push({
			tr: row,
			logs: logs
		});
	};

	obj.draw = function(){
		var i;
		//make sure transform is within bounds
		var t = zoom.translate(),
			tx = t[0],
			ty = t[1];

		var left_overshoot = x(beg) - margins.left;

		if (left_overshoot > 0)
			tx -= x(beg) - margins.left;

		var right_overshoot = x(end) - (plot_width+margins.left);
		if (right_overshoot < 0) {
			tx -= x(end) - (plot_width+margins.left);
		}
		zoom.translate([tx, ty]);

		/// get the data visible in the current window
		var window_beg = x.invert(margins.left),
			window_end = x.invert(margins.left+plot_width);

		var visible_data = [];
		for (i in datas) {
			var logs = datas[i];

			// get indicies before and after things in the window
			var beg_i = d3.bisectLeft( logs.map(function(l){return l.end;}), window_beg);
			var end_i = d3.bisectRight(logs.map(function(l){return l.beg;}), window_end, beg_i);
			beg_i = Math.max(0, beg_i-1);
			end_i = Math.min(logs.length, end_i+1);
			// get the data between these indicies
			var vis_logs = logs.slice(beg_i, end_i);
			visible_data.push(vis_logs);
		}
		var all_visible = d3.merge(visible_data);
		var val_extent = d3.extent(all_visible, function(l){return l[key];});
		var min_span = d3.min(all_visible, function(l){return l.end - l.beg;});

		// set y domain
		console.log(all_visible);
		console.log(val_extent);
		y.domain(val_extent);
		line.y(function(l){return y(l[key]);});

		// set zoom scale limits
		zoom.scaleExtent([1, (end-beg)/min_span]);
		xAxisG.call(xAxis);
		yAxisG.call(yAxis);

		for (i in paths) {
			paths[i].data([visible_data[i]]);
			paths[i].attr('d', line);
		}

		// diff plot only:
		if (rows.length == 2) {
			draw_background(visible_data);
		}
	};
	function draw_background(data) {
		o_logs = data[0];
		n_logs = data[1];

		var area_data = {
			same    : [],
			changed : [],
			added   : [],
			removed : []
		};

		function update_area_data(beg, end, cls) {

			if (area_data[cls][area_data[cls].length-1]) {
				area_data[cls][area_data[cls].length-1] = end;
			}
			else {
				area_data[cls].push(beg);
				area_data[cls].push(end);
			}
			// push null onto every other stack
			for (var k in area_data) {
				if (cls != k && area_data[k][area_data[k].length-1]) area_data[k].push(null);
			}
		}

		for (var i in o_logs) {
			var o_log = o_logs[i],
				n_log = n_logs[i];

			var cls = difference(o_log[key], n_log[key]);

			update_area_data(o_log.beg, o_log.end, cls);
		}
		area.x(x);
		for (var cls in area_data) {
			bg_areas[cls].data([area_data[cls]]);
			bg_areas[cls].attr('d', area);
		}
	}

	zoom.on('zoom', obj.draw);
	return obj;
}

// ---------------------------------------
function render_plot(parent, object) {
	load_csv(parent, data_id, object.data, function(logs) {
		var columns  = [],
			tooltips = [];

		for (var k in logs[0]){
			if (k == 'beg' || k == 'end') continue;
			columns.push(k);
			tooltips.push(logs[0][k]);
		}
		logs.shift();
		convert_dates(logs);
		// TODO: get variable name somehow
		var plt = plot(parent, columns, tooltips, object.var_name, object.ds_path);
		plt.add_line(logs, 'single');
		plt.add_row('', logs);
		plt.draw();
	});
}

// ---------------------------------------
// TODO: add background coloring to plot
function render_plot_diff(parent, object) {
	load_csv(parent, data_id, object.old_data, function(old_logs) {
		load_csv(parent, data_id, object.new_data, function(new_logs) {
			var columns = [],
				tooltips = [];

			for (var k in old_logs[0]) {
				if (k == 'beg' || k == 'end') continue;
				columns.push(k);
				tooltips.push(old_logs[0][k]);
			}
			old_logs.shift();
			new_logs.shift();

			convert_dates(old_logs);
			convert_dates(new_logs);

			// create plot
			var plt = plot(parent, columns, tooltips, object.var_name, 0, object.old_ds_path, object.new_ds_path);
			plt.add_background();
			plt.add_line(old_logs, 'old');
			plt.add_line(new_logs, 'new');
			plt.add_row('Old', old_logs);
			plt.add_row('new', new_logs);
			plt.draw();
		});
	});
}