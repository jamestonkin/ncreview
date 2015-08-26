#!/apps/base/python3/bin/python3
'''Command line interface to the datastream and datastreamdiff review modules.

Provides a command line interface to the functionality inside of the
datastream and datastreamdiff modules, which writes the resulting json string
to the user's ncreview_web directory.
'''

import os
import re
import csv
import sys
import time
import json
import errno
import argparse
import datetime as dt

import ncr.utils as utils
from ncr.datastream import Datastream
from ncr.datastreamdiff import DatastreamDiff

import pdb

### Progress Bar ------------------------------------------------------------------------------------------------------

class ProgressBar:
    '''Reports progress of loading datastreams, and estimates time remaining.
    '''
    def __init__(self, total_size, progress_bar_width=50):
        '''Initialize a progress bar

        Parameters:
            total_size             total number of files to process
            progress_bar_width  Character width of the progress bar
        '''
        # progress bar variables
        self.progress_bar_width = 50 # width of the progress bar in characters
        self.processed_size = 0
        self.total_size = total_size
        self.start_time = time.clock()

    def start(self):
        '''Save time when this method is called, and print a timeline at 0% progress.
        '''
        self.start_time = time.clock()
        sys.stdout.write('\r['+(' '*self.progress_bar_width)+']0%')

    def update(self, file_size):
        '''Increment the number of files processed by one, and update the progress bar accordingly
        '''
        ## update the progress bar
        self.processed_size += file_size
        time_elapsed = time.clock() - self.start_time
        total_time   = time_elapsed * self.total_size / self.processed_size
        time_remain  = (total_time - time_elapsed)*1.1 # overestimate by 10%

        mins_remain = int(time_remain // 60)
        secs_remain = int(time_remain % 60)

        progress = self.processed_size / self.total_size

        sys.stdout.write('\r[{0}{1}]{2}% ~{3}:{4} left    '.format(
            '#'*int(self.progress_bar_width*progress),
            ' '*int(self.progress_bar_width*(1-progress)),
            int(progress*100),
            mins_remain,
            "%02d" % secs_remain)
        )
        sys.stdout.flush()

    def complete(self):
        '''Display a progress bar at 100%
        '''
        print('\r['+('#'*self.progress_bar_width)+']100%'+' '*15)

### Parse Args --------------------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description='Review reprocessing between two directories, or summari',
    epilog='Note that if --begin and --end are unspecified when comparing datastreams, '+ \
    'the time span chosen will be the intersection of the time periods spanned by both datastreams.')
parser.add_argument('old_dir', help='Old datastream directory')

parser.add_argument('new_dir', nargs='?', default=None, 
    help='New datastream directory, exclude to simply review a single directory.')

parser.add_argument('--begin', '-b', default='00010101', metavar='YYYYMMDD', help='Ignore files before YYYYMMDD')
parser.add_argument('--end', '-e', default='99991231',  metavar='YYYYMMDD', help='Ignore files after YYYYMMDD')

parser.add_argument('--sample_interval', '-t', default=None,
	help='Time interval to average data over in HH-MM-SS. If not provided, '+\
         'defaults to 1 day if more than 10 days are being processed, otherwise defaults to hourly samples')

parser.add_argument('--metadata_only', '-m', action='store_true', default=False,
    help='Review only metadata, ignoring variable data. Much faster than standard review.')

parser.add_argument('--write_dir', '-w', default=None, metavar='DIR', help='write output data files to specified directory')

parser.add_argument('--name', '-n', default=None,
    help='Specify custom name to be used for the run.  Will be the directory name where the '+\
         'summary files ncreview creates are stored as well as the URL suffix.')

# convert time arguments 
args = parser.parse_args()

# get absolute directory paths,
# this will be important for the webpage to know where the datastreams came from
args.old_dir = os.path.abspath(args.old_dir)
if args.new_dir: args.new_dir = os.path.abspath(args.new_dir)
if args.write_dir: 
    args.write_dir = os.path.abspath(args.write_dir)

    if not os.path.exists(os.path.dirname(args.write_dir)):
        sys.stderr.write("Error: write directory %s does not exist\n"%os.path.dirname(args.write_dir))
        quit()

args.begin = dt.datetime.strptime(args.begin, '%Y%m%d')
args.end   = dt.datetime.strptime(args.end,   '%Y%m%d')

try:
    if args.sample_interval is not None:
        h, m, s = args.sample_interval.split('-')
        args.sample_interval = int(h)*60*60+int(m)*60+int(s)
except:
    sys.stderr.write("Error: chunk time %s is invalid.\n"%args.sample_interval)
    quit()

### Review Data -------------------------------------------------------------------------------------------------------
def is_valid(fname):
    t = utils.file_time(fname)
    return t is not None and args.begin <= t <= args.end

args.new_dir = os.path.abspath(args.new_dir) if args.new_dir else args.new_dir
args.old_dir = os.path.abspath(args.old_dir) if args.old_dir else args.old_dir

jdata = None
if args.new_dir:
    new_files = sorted(filter(is_valid, os.listdir(args.new_dir)))
    old_files = sorted(filter(is_valid, os.listdir(args.old_dir)))

    if not new_files:
        raise RuntimeError(args.new_dir+' contains no netCDF files in the specified time period.')
    if not old_files:
        raise RuntimeError(args.old_dir+' contains no netCDF files in the specified time period.')

    # get the latest begin and earliest end
    sys.stdout.write('Determining comparison interval...')
    new_times = list(map(utils.file_time, new_files))
    old_times = list(map(utils.file_time, old_files))

    args.begin = max(min(new_times), min(old_times)).replace(hour=0, minute=0, second=0, microsecond=0)
    args.end   = min(max(new_times), max(old_times)).replace(hour=23, minute=59, second=59, microsecond=999)

    # re-filter the files with the new time bounds
    new_files = sorted(filter(is_valid, new_files))
    old_files = sorted(filter(is_valid, old_files))

    if not new_files or not old_files:
        raise RuntimeError('Old and New directories do not appear to have overlapping measurement '+ \
            'times in the specified time period. Cannot determine a comparison interval.')
    sys.stdout.write('\r')

    sys.stdout.write('Determining total file size...'+' '*10)
    sys.stdout.flush()
    total_size = 0
    for f in old_files:
        total_size += os.stat(args.old_dir+'/'+f).st_size

    for f in new_files:
        total_size += os.stat(args.new_dir+'/'+f).st_size

    progress_bar = ProgressBar(total_size)
    sys.stdout.write('\r')
    # read datastreams
    print('Loading datastreams...'+' '*10)
    progress_bar.start()
    old_ds = Datastream(args.old_dir, args.begin, args.end, args.sample_interval, args.metadata_only, progress_bar)
    new_ds = Datastream(args.new_dir, args.begin, args.end, args.sample_interval, args.metadata_only, progress_bar)
    progress_bar.complete()
    # compare datastreams
    print('Comparing datastreams...')
    dsdiff = DatastreamDiff(old_ds, new_ds)
    jdata = dsdiff.jsonify()
else:
    path = args.old_dir

    files = sorted(filter(is_valid, os.listdir(path)))

    if not files:
        raise RuntimeError(path+' contains no netCDF files in the specified time period.')

    sys.stdout.write('Determining total file size...'+' '*10)
    sys.stdout.flush()
    total_size = 0
    for f in files:
        total_size += os.stat(path+'/'+f).st_size

    progress_bar = ProgressBar(total_size)
    sys.stdout.write('\r')
    # read datastreams
    print('Loading datastream...'+' '*10)
    progress_bar.start()
    ds = Datastream(path, args.begin, args.end, args.sample_interval, args.metadata_only, progress_bar)
    progress_bar.complete()

    jdata = ds.jsonify()

### Write out the data ------------------------------------------------------------------------------------------------

def unique_name(format_str, path):
    '''Produce a unique directory name at the specified path'''
    ID = 1
    while os.path.exists(path+'/'+format_str.format(ID)): ID += 1
    return format_str.format(ID)

# get the path of the dir to write to
wpath = '/data/tmp/ncreview/'

if args.write_dir is not None:
    wpath = args.write_dir

if not os.path.exists(wpath):
    os.mkdir(wpath)

format_str = ''
if args.name:
    format_str = args.name
    if os.path.exists(wpath+'/'+args.name): 
        format_str += '.{0}' # if the directory already exists, add a unique id

elif args.write_dir:
    format_str = '.ncr.'+dt.datetime.now().strftime('%y%m%d.%H%M%S')
    if os.path.exists(format_str): 
        format_str += '.{0}' # if the directory already exists, add a unique id
else:
    format_str = '%s.%s.{0}'%(os.environ['USER'], os.environ['HOST'])

jdata_dir = unique_name(format_str, wpath)

jdata_path = wpath+'/'+jdata_dir+'/'

os.mkdir(jdata_path)

n = 1
def separate_data(obj):
    global n
    to_separate = []
    if obj['type'] in ['plot', 'timeline', 'fileTimeline', 'timelineDiff']:
        to_separate = ['data']
    elif obj['type'] in ['plotDiff', 'fileTimelineDiff']:
        to_separate = ['old_data', 'new_data']

    for key in to_separate:
        # generate a unique csv file name
        while os.path.isfile(jdata_path+'ncreview.{0}.csv'.format(n)): n += 1

        # write out the data as csv
        with open(jdata_path+'ncreview.{0}.csv'.format(n), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, quoting=csv.QUOTE_NONNUMERIC)
            for row in obj[key]: writer.writerow(row)

        # make what was the data a reference to the file
        obj[key] = n

    if 'contents' in obj:
        for c in obj['contents']:
            separate_data(c)

separate_data(jdata)

# Write out the results as json
with open(jdata_path+'ncreview.json', 'w') as jfile:
    jfile.write(json.dumps(jdata, default=utils.JEncoder))

first_dir, user, *_ = os.path.realpath(__file__).split('/')[1:]
location = '/~'+user+'/dsutil' if first_dir == 'home' else ''

url_string = jdata_dir;

if args.write_dir: # if custom write location, put full path
    url_string = jdata_path

print ("Complete!")
print ("report at")
print ('https://engineering.arm.gov'+location+'/ncreview/?'+url_string)
