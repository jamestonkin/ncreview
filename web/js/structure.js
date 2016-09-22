/*
 * Maps type names to the functions that render that type of object
 */
var render = {
    'section': render_section,

    /* datastream types */
    'datastream'   : render_datastream,
    'variable'     : render_variable,
    'timeline'     : render_timeline,
    'fileTimeline' : render_file_timeline,
    'plot'         : render_plot, 
    'staticValue'  : render_static_value,
    'staticSummary': render_static_summary,

    /* datastream diff types */
    'datastreamDiff'   : render_datastream_diff,
    'groupDiff'        : render_group_diff,
    'variableDiff'     : render_variable_diff,
    'timelineDiff'     : render_timeline_diff,
    'fileTimelineDiff' : render_file_timeline_diff,
    'plotDiff'         : render_plot_diff,
    'staticValueDiff'  : render_static_value_diff,
    'staticSummaryDiff': render_static_summary_diff,
    'summary'          : render_summary,
};

/**
 * Generic object renderer. 
 * Calls the appropriate render function for the object according to its type.
 * Parameters:
 *      parent: d3 selection of the parent element in which to render contents
 *      object: object to render: eg. timeline, variable, plot, etc.
 *      data: top-level json structure.
 *              Was there a reason I didn't just make this a global variable??
 */
function render_object(parent, object) {
    if (!render.hasOwnProperty(object.type)) {
        throw new Error('Unrecognized object type: '+object.type);
    }
    console.log(object);
    render[object.type](parent, object);
}

/**
 * Renders an array of objects alphabetically sorted by name into the given parent element.
 * Loads the data needed to do this render only when nessecary.
 */
function render_contents(parent, contents) {
    // draw the contents, logging contents that couldn't be drawn
    // first, sort the contents into alphabetical order by name
    contents.sort(function(a, b){
        if (a.hasOwnProperty('name') && b.hasOwnProperty('name'))
            return (a.name > b.name) ? 1 : b.name > a.name ? -1 : 0;
        else
            return 0;
    });
    for (var i in contents){
        render_object(parent, contents[i]);
    }
}

/**
 * Generic section renderer
 * Creates a details element inside the given parent, and renders object.contents inside that details elem.
 * 
 */
function render_section(parent, object, summary_html) {
    if (summary_html === undefined) summary_html = object.name;
    var difference = 
        object.hasOwnProperty('difference') ? object.difference : 'same';
    var details = parent.append('details');
    var summary = details.append('summary')
        .attr('class', difference)
        .html(summary_html);
    details = details.append('div').style('padding-left', '1.25em');
    var loaded = false;
    summary.on('click', function(){
        if (loaded) return;
        render_contents(details, object.contents);
        loaded = true;
    });
}

/**
 * Generic section renderer
 * modified from original to use a custom details structure, that is compatible on other browsers
 *
 */
/*
function render_section(parent, object, summary_html) {
    if (summary_html === undefined) summary_html = object.name;
    var difference = 
        object.hasOwnProperty('difference') ? object.difference : 'same';
    var summary = parent.append('div').attr('class', 'summary');
    var details = summary.append('div').attr('class', 'detals');

}
*/
/**
 * Creates a details element and summary with counts of same, added, and changed.
 */
function render_group_diff(parent, object) {
    var html = object.name;
    var n_diffs = object.n_diffs;
    n_diffs_array = [];
    diff_types = ['same', 'changed', 'removed', 'added'];
    for (var i in diff_types) {
        var d = diff_types[i];
        if (n_diffs[d]) {
            n_diffs_array.push(n_diffs[d]+' '+d);
        }
    }

    // summary of same, added, removed, and changed
    if (n_diffs_array.length)
        html += ' <i>('+n_diffs_array.join(', ')+')</i>';

    render_section(parent, object, html);
}