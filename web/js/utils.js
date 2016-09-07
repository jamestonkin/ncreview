// utilities

margins = {
    top   : 10,
    bottom: 30,
    left  : 80,
    right : 120
};

/**
 * Compare values and return their status as same/changed/etc.
 * if a single value is given, it is assumed to be a diff and arg.old and arg.new are compared
 */
function difference(o, n) {
	if (n !== undefined)
		return o == n ? 'same' :
			   o === null || o == '' ? 'added' :
			   n === null || n == '' ? 'removed' :
			   'changed';
	else
		return difference(o.old, o.new);
}

format = d3.time.format('%Y%m%d.%H%M%S');

/**
 * Get the average of two times
 */
function mean_time(diff) {
	return new Date((diff.beg.getTime()+diff.end.getTime())/2);
}

/**
 * Convert seconds since epoch to a psuedo-utc date, by timezoneoffset to the date.
 */
function epoch2utc(epoch) {
	d = new Date(epoch * 1000);
	return new Date(d.getTime() + d.getTimezoneOffset()*60*1000);
}

/**
 * Returns a rect (aka log, diff, etc.) in rects such that rect.beg <= x0 <= rect.end
 * If none are found, return null
 */
function get_rect_at(x0, rects) {
	var rect = null;
	//var min_dist = Infinity;
	for (var i=0; i < rects.length; i++) {
		if (rects[i].beg <= x0 && x0 <= rects[i].end) {
			rect = rects[i];
			break;
		}
	}
	return rect;
}

/**
 * Returns a rect (aka log, diff, etc.) in rects such that rect.beg <= x0 <= rect.end
 * If none are found, returns the rect closest to x0.
 */
function get_rect_near(x0, rects) {
	var rect = null;
	var min_dist = Infinity;
	for (var i=0; i < rects.length; i++) {
		if (rects[i].beg <= x0 && x0 <= rects[i].end) {
			rect = rects[i];
			break;
		}
		dist = Math.min(
			Math.abs(rects[i].beg-x0),
			Math.abs(rects[i].end-x0)
			);
		if (dist < min_dist) {
			min_dist = dist;
			rect = rects[i];
		}
	}
	return rect;
}

function select_i(selection, n) {
	return selection.filter(function(d, i){return i==n;});
}