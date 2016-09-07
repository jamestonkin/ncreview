function dims_html(dims) {
    var html = ' <i>(';
    if (dims !== 'varying')
        html += dims.join(', ');
    else
        html += 'varying dimensions';
    html += ')</i>';
    return html;
}

function render_variable(parent, object){
	var html = object.name;
    html += dims_html(object.dims);
    render_section(parent, object, html);
}

function render_variable_diff(parent, object){
	var html = object.name;
    html += dims_html(object.dims);
    render_section(parent, object, html);
}
