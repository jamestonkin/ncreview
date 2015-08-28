#Guide to ncreview

`ncreview` is a tool which allows users to produce interactive web-based comparisons between datastreams or summaries of a single datastream, providing information on netCDF data and metadata. The metadata part of the review is produced in a non-lossy way which preserves all metadata information present throughout each datastream. Numerical data is summarized with statistics like min, max, mean, n_missing, etc. for a summary interval which can be specified by the user at the command line.

##Setup
To use ncreview normally on the ARM servers, there are a few modifications you need to make to your enviornment.
Set the following enviornment variables in your profile:
 - `PATH` to `/apps/ds/bin:$PATH`
 - `PYTHONPATH` to `/apps/ds/lib`

##Command Line Interface

The reports are created through the `ncreview` command line interface, and deposited at a URL to be opened in a browser. Usage help can be found by typing `ncreview --help`.

##Web Report

The web-based report is laid out in a heirarchical structure of nested expandable elements, which, for comparisons, are color-coded to indicate the difference in the data they contain. As described in the expandable legend in the upper-right hand corner, throughout the report blue is used to indicate that data has changed, red is used to indicate that data was in the old report but is not in the new (removed), and green means that data is in the new report but not the old one (added).

At the top level, there are four main sections into which the summary information is divided:

###File Timeline

 The file timeline provides a visual summary of what time periods files in the datastream cover. The timeline is interactive, and users can zoom and pan around it using the mouse. Hovering over one of the grey file rectangles will bring up its begin and end times in the table below.

###Attributes

 The attributes section provides summary information on global attributes throughout the datastream(s). If an attribute's value remained constant over the collection of files scanned, it will be displayed as a static value next to its name. If the attributes value varied, or was even different in just one file, that value will be displayed on a timeline. This timeline works very similarly to the file timeline: it can be zoomed and panned, and hovering over some section of the timeline reveals the attribute's value at that time.

###Dimensions

 The dimensions section works very similarly to the attributes section, but instead of displaying attribute names and values, it displays dimensions names and lengths.

###Variables

Each variable section contains a summary of its data, a list of its dimensions, and a variable attributes section which behaves precisely like the global attributes section. If the variable has companion variables such as QC data, these will be stored in a companions section in the variable structure.

####Data

A variable's data can be displayed in several formats, depending on its dimensionality:

- **Dimensionless**

    A dimensionless variable's data is displayed as either a static value or as a timeline, just like dimension lengths or attribute values.

- **Dimensioned by `time`**
    
    Data dimensioned by `time` is displayed in an interactive plot which plots one of a number of summary statistics. To change which summary statistic is being displayed, click the name of that statistic in the table below the plot.

    When making a comparison, a background color appears behind the plot lines when the data differs, according to the color scheme in the legend. 

    The dq inspector plots button below the data summary will generate a Pytdq_inspector plot of the data for time range specified by the bounds of the interactive plot. This can be useful when zooming in to produce higher detail plots than are available interactively, or to leverage any of the extra functionality provided by the dq_inspector tool. Be weary, however, that dq_inspector plots can take a while to appear for large sets of data

- **Dimensioned, but not by `time`**

    In this case, each file's data is summarized into a few values like min, max and mean, and these values are displayed in a table. If values vary from file to file, then they are displayed in an interactive plot, which works very similarly to the timed plot.
    

