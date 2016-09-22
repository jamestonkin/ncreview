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
import math
import time
import json
import errno
import signal
import argparse
import traceback
import datetime as dt
from concurrent.futures import ProcessPoolExecutor

import ncr.utils as utils
from ncr.summary import SumFile
from ncr.datastream import Datastream
from ncr.datastreamdiff import DatastreamDiff

# Progress Bar -----------------------------------------------------------


class ProgressBar:
    '''Reports progress of loading datastreams, and estimates time remaining.
    '''

    def __init__(self, total, width=50):
        '''Initialize a progress bar
        Parameters:
            total  numeric value of what "complete" represents
            width  character width of the progress bar
        '''
        self.width = width
        self.done = 0
        self.total = total
        self.started = time.perf_counter()

    def start(self):
        '''Save time when this method is called, and print a timeline at 0% progress.
        '''
        self.started = time.perf_counter()
        sys.stdout.write('\r[' + (' ' * self.width) + ']0%')

    def update(self, amount):
        '''Increment the number of files processed by one, and update the progress bar accordingly
        '''
        self.done += amount
        elapsed = time.perf_counter() - self.started
        estimate = elapsed * self.total / self.done
        remains = (estimate - elapsed) * 1.1  # overestimate by 10%
        progress = self.done / self.total
        sys.stdout.write('\r[{0}{1}]{2}% ~{3} left    '.format(
            '#' * int(self.width * progress),
            ' ' * int(self.width * (1 - progress)),
            int(progress * 100),
            utils.time_diff(int(remains)))
        )
        sys.stdout.flush()

    def complete(self):
        '''Display a progress bar at 100%
        '''
        print('\r[' + ('#' * self.width) + ']100%' + ' ' * 20)

# Utilities --------------------------------------------------------------


def is_plottable(path):
    path_match = re.search('\/([a-z]{3})\/\\1[a-zA-Z0-9\.]+\s*$', path)
    return path_match is not None


def summarize_all(root, paths, interval, mdonly):
    for p in paths:
        yield summarize(('%s/%s' % (root, p), interval, mdonly))


def summarize(args):
    (path, interval, mdonly) = args
    f = SumFile(path, interval=interval, mdonly=mdonly)
    f.read()
    return f.jsonify()

# Main -------------------------------------------------------------------


def main(argv):

    start = time.time()

    min_readers = 1
    max_readers = 20

    # NOTE: the defaults for begin time and end time
    # Begin : 00010101
    # End : 99991231

    # Parse Args -------------------------------------------------------------

    parser = argparse.ArgumentParser(description='Compare netCDF files between two directories or summarize from single directory',
                                     epilog='''Note that if --begin and --end are unspecified when comparing datastreams,
    the time span chosen will be the intersection of the time periods spanned by both datastreams.''')
    parser.add_argument('old_dir', help='Old netCDF files directory')

    parser.add_argument('new_dir', nargs='?', default=None,
                        help='New netCDF files directory, exclude to simply summarize a single directory.')

    parser.add_argument('--begin', '-b', default='00010101',
                        metavar='YYYYMMDD', help='Ignore files before YYYYMMDD')
    parser.add_argument('--end', '-e', default='99991231',
                        metavar='YYYYMMDD', help='Ignore files after YYYYMMDD')

    parser.add_argument('--sample_interval', '-t', default=None,
                        help='Time interval to average data over in HH-MM-SS. If not provided, ' +
                        'defaults to 1 day if more than 10 days are being processed, otherwise defaults to hourly samples')

    parser.add_argument('--metadata_only', '-m', action='store_true', default=False,
                        help='Review only metadata, ignoring variable data. Much faster than standard review.')

    parser.add_argument('--write_dir', '-w', default=None, metavar='DIR',
                        help='write output data files to specified directory')

    parser.add_argument('--name', '-n', default=None,
                        help='Specify custom name to be used for the run.  Will be the directory name where the ' +
                        'summary files ncreview creates are stored as well as the URL suffix.')

    parser.add_argument('--readers', type=int, default=10,
                        help='Specify number of concurrent file readers.  Will accept a number between %d and %d (inclusive).' % (min_readers, max_readers))

    # edit
    parser.add_argument('--dev', action='store_true',
                        help='For running on a local host, dev. only')

    args = parser.parse_args()

    # edit
    global DEVELOPMENT
    DEVELOPMENT = args.dev

        # Get absolute directory paths...
        # This will be important for the webpage to know where the datastreams
        # came from.
    args.old_dir = os.path.abspath(args.old_dir)
    if args.new_dir:
        args.new_dir = os.path.abspath(args.new_dir)
    if args.write_dir:
        args.write_dir = os.path.abspath(args.write_dir)

        if not os.path.exists(os.path.dirname(args.write_dir)):
            raise ValueError(
                "Error: write directory %s does not exist\n" % os.path.dirname(args.write_dir))

    args.begin = dt.datetime.strptime(args.begin, '%Y%m%d')
    args.end = dt.datetime.strptime(args.end,   '%Y%m%d')

    if args.readers < min_readers or args.readers > max_readers:
        raise ValueError("Error: number of readers must be between %d and %d (inclusive)." % (
            min_readers, max_readers))

    try:
        if args.sample_interval is not None:
            h, m, s = args.sample_interval.split('-')
            args.sample_interval = int(h) * 60 * 60 + int(m) * 60 + int(s)
        # if interval is more than 10 days
        elif args.end - args.begin > dt.timedelta(days=10):
            args.sample_interval = 24 * 60 * 60  # set interval to 24 hr
        else:
            args.sample_interval = 60 * 60  # set interval to 1 hr
    except:
        raise ValueError("Error: chunk time %s is invalid.\n" %
                         args.sample_interval)

    if args.sample_interval <= 0:  # if user specified non-positive sample interval
        raise ValueError(
            'Error: sample interval must be a positive number, not ' + str(args.sample_interval))

    # Review Data ------------------------------------------------------------

    def is_valid(fname):
        t = utils.file_time(fname)
        return t is not None and args.begin <= t <= args.end

    args.new_dir = os.path.abspath(
        args.new_dir) if args.new_dir else args.new_dir
    args.old_dir = os.path.abspath(
        args.old_dir) if args.old_dir else args.old_dir

    jdata = None
    if args.new_dir:
        new_files = sorted(filter(is_valid, os.listdir(args.new_dir)))
        old_files = sorted(filter(is_valid, os.listdir(args.old_dir)))

        if not new_files:
            raise RuntimeError(
                args.new_dir + ' contains no netCDF files in the specified time period.')
        if not old_files:
            raise RuntimeError(
                args.new_dir + ' contains no netCDF files in the specified time period.')

        # Get the latest begin and earliest end
        new_times = list(map(utils.file_time, new_files))
        old_times = list(map(utils.file_time, old_files))

        # These values are hardcoded to match the default dates
        # If user passed in start/end times, show the entire timeline at those dates
        # Otherwise the program defaults to only showing overlap
        if str(args.begin) != '0001-01-01 00:00:00':
            args.begin = min(min(new_times), min(old_times)).replace(
            hour=0, minute=0, second=0, microsecond=0)
        else:
            args.begin = max(min(new_times), min(old_times)).replace(
            hour=0, minute=0, second=0, microsecond=0)

        if str(args.end) != '9999-12-31 00:00:00':
            args.end = max(max(new_times), max(old_times)).replace(
            hour=23, minute=59, second=59, microsecond=999)
            pass
        else:
            args.end = min(max(new_times), max(old_times)).replace(
            hour=23, minute=59, second=59, microsecond=999)

        # Re-filter the files with the new time bounds
        new_files = sorted(filter(is_valid, new_files))
        old_files = sorted(filter(is_valid, old_files))

        if not new_files or not old_files:
            raise RuntimeError('Old and New directories do not appear to have overlapping measurement ' +
                               'times in the specified time period. Cannot determine a comparison interval.')

        print('Scanning directories...')

        total_size = 0
        for (which, path, files) in (('old', args.old_dir, old_files), ('new', args.new_dir, new_files)):
            for f in files:
                total_size += os.stat('%s/%s' % (path, f)).st_size

        progress_bar = ProgressBar(total_size)

        print('Reading data...')

        old_ds = Datastream(is_plottable(args.old_dir), args.sample_interval)
        new_ds = Datastream(is_plottable(args.new_dir), args.sample_interval)

        progress_bar.start()

        with ProcessPoolExecutor(max_workers=args.readers) as executor:
            for s in executor.map(summarize, map(lambda f: ('%s/%s' % (args.old_dir, f), args.sample_interval, args.metadata_only), old_files)):
                old_ds.add(s)
                progress_bar.update(os.stat(s['path']).st_size)
            for s in executor.map(summarize, map(lambda f: ('%s/%s' % (args.new_dir, f), args.sample_interval, args.metadata_only), new_files)):
                new_ds.add(s)
                progress_bar.update(os.stat(s['path']).st_size)

        progress_bar.complete()

        print('Comparing...')
        dsdiff = DatastreamDiff(old_ds, new_ds)
        jdata = dsdiff.jsonify()

    else:
        path = args.old_dir

        files = sorted(filter(is_valid, os.listdir(path)))

        if not files:
            raise RuntimeError(
                path + ' contains no netCDF files in the specified time period.')

        print('Scanning directory...')

        total_size = 0
        for f in files:
            total_size += os.stat('%s/%s' % (path, f)).st_size

        progress_bar = ProgressBar(total_size)

        print('Reading data...')
        ds = Datastream(is_plottable(path), args.sample_interval)

        progress_bar.start()

        with ProcessPoolExecutor(max_workers=args.readers) as executor:
            for s in executor.map(summarize, map(lambda f: ('%s/%s' % (path, f), args.sample_interval, args.metadata_only), files)):
                ds.add(s)
                progress_bar.update(os.stat(s['path']).st_size)

        progress_bar.complete()

        jdata = ds.jsonify()

    # Write out the data -----------------------------------------------------

    def unique_name(format_str, path):
        '''Produce a unique directory name at the specified path'''
        ID = 1
        while os.path.exists(path + '/' + format_str.format(ID)):
            ID += 1
        return format_str.format(ID)

    wpath = '/data/tmp/ncreview/'

    if args.write_dir is not None:
        wpath = args.write_dir

    if not os.path.exists(wpath):
        os.mkdir(wpath)

    format_str = ''
    if args.name:
        format_str = args.name
        if os.path.exists(wpath + '/' + args.name):
            # if the directory already exists, add a unique id
            format_str += '.{0}'

    elif args.write_dir:
        format_str = '.ncr.' + dt.datetime.now().strftime('%y%m%d.%H%M%S')
        if os.path.exists(format_str):
            # if the directory already exists, add a unique id
            format_str += '.{0}'
    else:
        format_str = '%s.%s.{0}' % (os.environ['USER'], os.environ['HOST'])

    jdata_dir = unique_name(format_str, wpath)

    jdata_path = wpath + '/' + jdata_dir + '/'
    os.mkdir(jdata_path)

    def separate_data(obj, n=1):
        to_separate = []
        if obj['type'] in ['plot', 'timeline', 'fileTimeline', 'timelineDiff']:
            to_separate = ['data']
        elif obj['type'] in ['plotDiff', 'fileTimelineDiff']:
            to_separate = ['old_data', 'new_data']

        for key in to_separate:
            # Generate a unique csv file name
            while os.path.isfile(jdata_path + 'ncreview.{0}.csv'.format(n)):
                n += 1

            # Write out the data as csv
            with open(jdata_path + 'ncreview.{0}.csv'.format(n), 'w', newline='') as csvfile:
                writer = csv.writer(csvfile, quoting=csv.QUOTE_NONNUMERIC)
                for row in obj[key]:
                    writer.writerow(row)

            # Make what was the data a reference to the file
            obj[key] = n

        if 'contents' in obj:
            for c in obj['contents']:
                separate_data(c, n)

    separate_data(jdata)

    with open(jdata_path + 'ncreview.json', 'w') as jfile:
        jfile.write(json.dumps(jdata, default=utils.JEncoder))

    first_dir, user, *_ = os.path.realpath(__file__).split('/')[1:]
    location = '/~' + user + '/dsutil' if first_dir == 'home' else ''

    url_string = jdata_dir

    if args.write_dir:  # if custom write location, put full path
        url_string = jdata_path

    # reads csv file
    # finds new/old values that are different
    """different_times = []
                            # the .5.csv is the file that has
                            file_path = '/data/tmp/ncreview/' + url_string + '/ncreview.5.csv'
                            with open(file_path, newline='') as csvfile:
                                reading = csv.reader(csvfile, delimiter = ',')
                                counter = 0
                                for row in reading:
                                    old, new, begin, end = row
                                    #dont need end
                                    if counter == 0:
                                        counter += 1
                                        continue
                                    if old != new:
                                        #date = time.ctime(int(begin))
                                        date = dt.datetime.fromtimestamp(int(begin)).strftime('%Y-%m-%d')
                                        print('date: {}\told: {}\tnew: {}\tdifference: {}'.format(date, old, new, int(new) - int(old)))"""

    print("")
    print("Complete! Took %s." % utils.time_diff(time.time() - start))
    print("------------------------------------------------------------------------")
    print('https://engineering.arm.gov' +
          location + '/ncreview/?' + url_string)
    print("")

    if DEVELOPMENT:
        print('DEVELOPMENT LINK: \n' + 'localhost/web/index.html?' + url_string + '\n\n')

    return 0


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except Exception as e:
        sys.stderr.write(e)
        sys.exit(1)
    sys.exit(0)
