/**
 * Returns formatted HTML of some value to display;
 * truncating it if it's longer than max_chars.
 */
function val_html(val, max_chars) {
    html = '';
    if (typeof val == 'string' && val.length > max_chars) {
        html += val.slice(0, max_chars)+'<span class="ellipsis">...</span>';
        html += '<span class="remaining" hidden>';
        html += val.slice(max_chars, val.length)+'</span>';
        truncated = true;
    }
    else if (val.length !== undefined && !val.length) {
        html += '<i>none</i>';
    }
    else
        html += val;

    return html;
}

/**
 * Renders a single-line static key-value pair, common for attributes and dimensions.
 */
function render_static_value(parent, object) {
    var cls = 
        object.hasOwnProperty('difference') ? object.difference : 'same';

    var html = '<span class="'+cls+'"">'+object.name+':</span> ';

    var truncated = true;
    html += val_html(object.val, 64);

    var div = parent.append('div').html(html).attr('class', 'staticValue');

    if (truncated) {
        div.on('click', function(){
            if (truncated) {
                div.select('.ellipsis').attr('hidden', '');
                div.select('.remaining').attr('hidden', null);
            }
            else {
                div.select('.ellipsis').attr('hidden', null);
                div.select('.remaining').attr('hidden', '');
            }
            truncated = !truncated;
        });
    }
}

/**
 * Renders a comparison of old and new static values.
 */
function render_static_value_diff(parent, object) {
    var cls = 
        object.hasOwnProperty('difference') ? object.difference : 'same';

    var html = '<span class="'+cls+'"">'+object.name+':</span> ';

    var truncated = true;

    // add the old value
    html += '<br hidden>' +
            '<span style="font-weight:bold"> Old: </span>' +
            val_html(object.old, 32);
    html +='<wbr>';
    html +='<br hidden>' +
           '<span style="font-weight:bold"> New: </span>' +
            val_html(object.new, 32);

    var div = parent.append('div').html(html).attr('class', 'staticValue');

    div.on('click', function(){
        if (truncated) {
            div.selectAll('br').attr('hidden', null);
            div.selectAll('.ellipsis').attr('hidden', '');
            div.selectAll('.remaining').attr('hidden', null);
        }
        else {
            div.selectAll('br').attr('hidden', '');
            div.selectAll('.ellipsis').attr('hidden', null);
            div.selectAll('.remaining').attr('hidden', '');
        }
        truncated = !truncated;
    });
}

/**
 * Render a variable summary when there is a single summary object to report.
 */
function render_static_summary(parent, object) {
    var difference = object.hasOwnProperty('difference') ? object.difference : 'same';
    var details = parent.append('details');
    var summary = details.append('summary')
        .attr('class', difference)
        .html(object.name);
    details = details.append('div').style('padding-left', '1.25em');
    var table = details.append('table');
    var hr = table.append('tr');
    var dr = table.append('tr');
    for (var i in object.columns) {
        hr.append('th')
            .text(object.columns[i])
            .attr('title', object.tooltips[i]);
        dr.append('td').html(object.val[i] !== null ? object.val[i] : '<i>N/A</i>');
    }
}

/**
 * Render a comparison of old and new static variable summaries.
 */
function render_static_summary_diff(parent, object) {
    if (object.data_type == 'dimless') {
        render_static_value_diff(parent, object);
        return;
    }

    var diff_cls = object.hasOwnProperty('difference') ? object.difference : 'same';
    var details = parent.append('details');
    var summary = details.append('summary')
        .attr('class', diff_cls)
        .html(object.name);
    details = details.append('div').style('padding-left', '1.25em');
    var table = details.append('table');
    var hr = table.append('tr');
    var or = table.append('tr');
    var nr = table.append('tr');

    hr.append('td');
    or.append('th').text('Old');
    nr.append('th').text('New');

    for (var i in object.columns) {
        var cls = difference(object.old[i], object.new[i]);
        hr.append('th')
            .text(object.columns[i])
            .attr('title', object.tooltips[i])
            .attr('class', cls);
        or.append('td').html(object.old[i] !== null ? object.old[i] : '<i>N/A</i>');
        nr.append('td').html(object.new[i] !== null ? object.new[i] : '<i>N/A</i>');
    }
}
