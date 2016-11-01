'''Datastream reader for the ncreview tool.

This contains the classes nessecary to read a single datastream into a
Datastream object, collecting useful summary information about the datastream
which can be compared against another datastream with datastreamdiff, or
written out as a JSON string for display on the ncreview web page.

Recurring Attributes:
A '.name' attribute is the name of the object's corresponding section in the
    web-based UI

A '.ds' attribute refers back to the Datastream that contains the object.

Recurring methods:

A .load(data) method is used to update a summary object with a single file worth of data.
    the data parameter takes whatever data from is needed for the summary

A .jsonify() method returns a data structure that can be converted to json
    and used to generate the report in the web-based UI.
'''

import os
import re
import sys
import time
import json
import logging
import traceback
import numpy as np
import datetime as dt
from netCDF4 import Dataset
from collections import namedtuple

import ncr.utils as utils

### Timeline ----------------------------------------------------------------------------------------------------------

Log = namedtuple('Log', ['val', 'beg', 'end'])

class Timeline(list):
    '''A record of some data in the datastream which changes over time.

    A Timeline, which extends the builtin Python list, stores a sequence of 
    logs which provide a comprehensive history of whatever value the timeline 
    may be recording (an attribute value, a dimension length, etc.) in the 
    datastream. 
    beg and end in a Log are indicies of the first and last file 
    in self.ds.file_timeline for which recorded value was equal to val.
    '''
    def __init__(self, name, ds):
        super(Timeline, self).__init__(self)
        self.name = name
        self.ds   = ds

    def load(self, val):
        '''
        Parameters:
        val: Any object which can be tested for equality
        '''
        fi = len(self.ds.file_timeline)-1

        if self and self[-1].val == val and self[-1].end + 1 == fi:
            self[-1] = Log(self[-1].val, self[-1].beg, fi)
        else:
            self.append(Log(val, fi, fi))

    def jsonify(self):
        if len(self) == 1:
            sec = {
                'type': 'staticValue',
                'name': self.name,
                'val': self[0].val
            }
            if hasattr(self, '_difference'):
                sec['difference'] = self._difference
            return sec
        else:
            return utils.json_section(self, 
                [
                    {
                        'type': 'timeline',
                        'data': [['beg', 'end', 'val']]+[
                            [
                                self.ds.file_timeline[l.beg].beg,
                                self.ds.file_timeline[l.end].end,
                                l.val
                            ] for l in self
                        ],
                    }
                ])
            
### Summaries ---------------------------------------------------------------------------------------------------------

class DimlessSum:
    '''Summary of dimensionless data.

    A "summary" of data in a dimensionless variable. Simply wraps the
    dimensionless value with the methods expected of a data summary.
    '''
    def __init__(self, val):
        self.val = val

    def __eq__(self, other):
        return isinstance(other, DimlessSum) and self.val == other.val

    def __ne__(self, other):
        return not self.__eq__(other)

    def row(self):
        return [self.val]

class NumSum:
    '''Summary of the values in a variable holding numerical data.

    This numerical summary (NumSum) holds summary statistics of a dataset. the +=
    operator correctly combines two statistical summaries, such that for
    datasets a and b, numsum(a)+numsum(b) = numsum(a concatenated with b).
    '''
    __slots__ = \
        ('n', 'ngood', 'nmiss', 'ninf', 'nnan', 'nfill', 'min', 'max', 'mean', 'var', 'std')

    def __init__(self,
            n      = 0,
            ngood  = 0,
            nmiss  = 0,
            ninf   = 0,
            nnan   = 0,
            nfill  = 0,
            min    = None,
            max    = None,
            mean   = None,
            median = None,
            var    = None,
            std    = None
            ):
        self.n      = n
        self.ngood  = ngood
        self.nmiss  = nmiss
        self.ninf   = ninf
        self.nnan   = nnan
        self.nfill  = nfill
        self.min    = min
        self.max    = max
        self.mean   = mean
        self.var    = var
        self.std    = std

    def get_nifs(self):
        return self.nmiss, self.nnan, self.ninf, self.nfill

    def __iadd__(self, other):
        s, o = self, other # for brevity
        ngood = s.ngood + o.ngood
        try:
            mean = s.mean if o.mean is None else \
                   o.mean if s.mean is None else \
                   (s.mean*s.ngood + o.mean*o.ngood)/ngood \
                   if ngood > 0 else None
        except TypeError:
            mean = None

        try:
            if o.var is None: pass
            elif s.var is None:
                s.var = o.var
            elif ngood > 0 and mean is not None:
                # http://stats.stackexchange.com/questions/43159/ \
                # how-to-calculate-pooled-variance-of-two-groups-given-known-group-variances-mean
                # NOTE: I added the abs() because you can never end up with a negative variance and apparently
                # sometimes if the number is really close to zero you might end up with it looking negative.
                s.var = abs( (s.ngood * (s.var+s.mean**2) + o.ngood * (o.var+o.mean**2)) / ngood - mean**2 )
            else:
                s.var = None
        except TypeError:
            s.var = None

        s.n     = s.n + o.n
        s.ngood = ngood
        s.nmiss = s.nmiss + o.nmiss
        s.nnan  = s.nnan  + o.nnan
        s.ninf  = s.ninf  + o.ninf
        s.nfill = s.nfill + o.nfill

        if s.min is None and o.min is not None: s.min = o.min
        elif s.min is not None and o.min is not None: s.min = min(s.min, o.min)

        if s.max is None and o.max is not None: s.max = o.max
        elif s.max is not None and o.max is not None: s.max = max(s.max, o.max)

        s.mean  = mean
        return s
    
    def __eq__(self, other):
        '''Tests the equivalence of two numerical summaries to within plausible rounding errors'''
        if not isinstance(other, NumSum):
            return False
        for att in NumSum.__slots__:
            s = getattr(self, att)
            o = getattr(other, att)
            if s is None and o is None: continue
            if s is None or o is None: return False
            if np.abs(s - o) > 1e-4: return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def row(self):
        '''Converts the summary into a list for storage in a table as a row'''
        return [
            self.n,
            self.ngood,
            self.nmiss,
            self.ninf,
            self.nnan,
            self.nfill,
            self.min,
            self.max,
            self.mean,
            np.sqrt(self.var) if self.var is not None and self.var >= 0 else None
        ]

    def jsonify(self):
        data = {
            'n'      : self.n,
            'ngood'  : self.ngood,
            'nmiss'  : self.nmiss,
            'ninf'   : self.ninf,
            'nnan'   : self.nnan,
            'min'    : self.min,
            'max'    : self.max,
            'mean'   : self.mean,
            'std'    : np.sqrt(self.var) if self.var is not None and self.var >= 0 else None
        }
        return data

class StateSum(list):
    '''Summary of a state indicator variable.

    Holds an array with the number of occurrences of each state in a state 
    indicator variable. the += operator combines two state summaries by the 
    second operand's flag counts to the first.
    '''
    def __init__(self, vals):
        list.__init__(self)
        self._s = list(map(lambda x: x[0], vals))
        self._m = { s: i for i,s in enumerate(self._s) }
        self[:] = list(map(lambda x: int(x[1]), vals))

    def __iadd__(self, other):
        for i,x in enumerate(other):
            self.add(other.state(i), x)

        return self

    def __eq__(self, other):
        return isinstance(other, StateSum) and\
               len(self) == len(other)     and\
               all(a == b for a,b in zip(self, other)) and\
               all(a == b for a,b in zip(self.states(), other.states()))

    def __ne__(self, other):
        return not self.__eq__(other)

    def get_nifs(self):
        return (0,0,0,0)

    def state(self, i):
        return self._s[i]

    def states(self):
        return self._s

    def add(self, s, x):
        if s not in self._m:
            self._m[s] = len(self._s)
            self._s.append(s)
        self[self._m[s]] += x

    def row(self):
        return self[:]

### Data Types --------------------------------------------------------------------------------------------------------

class DimlessDataType:
    '''Wraps a single value with the expected DataType interface.
    '''
    def __init__(self, sumvar, ds):
        self.type_name = 'numericScalar'

    @staticmethod
    def matches(sumvar, ds):
        return not sumvar['dimensions']

    def summarize(self, data=None):
        return DimlessSum(data) if data is not None else DimlessSum(None)

    def columns(self):
        return ['value'], ['value']


class NumDataType:
    '''Class containing methods for working with numerical data

    Performs numerical summary of data and returns the result as a NumSum
    '''
    def __init__(self, sumvar, ds):
        self.type_name = 'numericSeries'

    @staticmethod
    def matches(sumvar, ds):
        return True

    def summarize(self, data=None):
        '''Return summary statistics of the data as a NumSum object

        parameters:
        data: dict of numeric summaries to be wrapped in a NumSum instance
        '''
        if data is None:
            return NumSum()
        return NumSum(**data)

    def columns(self):
        cols = ['n', 'ngood', 'nmiss', 'ninf', 'nnan', 'nfill', 'min', 'max', 'mean', 'std']
        tooltips = [
            'Number of samples',
            'Number of good samples',
            'Number of missing samples',
            'Number of infs',
            'Number of nans',
            'Number of fill values',
            'Minimum value',
            'Maximum value',
            'Mean value',
            'Median value',
            'Standard deviation']

        return cols, tooltips


class ExStateDataType:
    '''Class containig methods for working with exclusive state data.

    Reads important metadata from an exclusive state variable, and collects
    the counts of each exclusive state in a data set into a StateSum.
    '''
    flag_desc_re = re.compile('^flag_(\d+)_description$')
    def __init__(self, sumvar, ds):
        self.type_name = 'stateExclusive'
        self.flag_values = []
        self.attrs = {a['name']:a['val'] for a in sumvar['attributes']}
        if 'flag_values' in self.attrs:
            self.flag_values = self.attrs['flag_values']
        else:
            n_flags = 0
            for a,v in self.attrs.items():
                match = ExStateDataType.flag_desc_re.match(a)
                if not match: continue
                n_flags = max(n_flags, int(match.group(1)))
            self.flag_values = range(1, n_flags+1)
        self.flag_descriptions = ['']*len(self.flag_values)
        for n in range(len(self.flag_values)):
            # Allow for gaps in numbers...
            flag_num = self.flag_values[n]
            flag_desc = 'flag_'+str(flag_num+1)+'_description'
            if flag_desc in self.attrs:
                self.flag_descriptions[n] = self.attrs[flag_desc]
            elif 'flag_meanings' in self.attrs:
                self.flag_descriptions[n] = self.attrs['flag_meanings'].split()[n]

    def columns(self):
        return ['flag '+str(v) for v in self.flag_values], self.flag_descriptions

    @staticmethod
    def matches(sumvar, ds):
        '''
        Parameters
        sumvar: summary variable dict to check
        ds: parent Datastream
        '''
        return sumvar['dtype'][:3] == 'int' and (
            any(a['name'] == 'flag_values' for a in sumvar['attributes']) or \
            any(ExStateDataType.flag_desc_re.match(a['name']) for a in sumvar['attributes'])
            )
                
    def summarize(self, data=None):
        if not data:
            return StateSum([(str(v),0) for v in self.flag_values])
        return StateSum([ (str(v), data[str(v)] if str(v) in data else 0) for v in self.flag_values ])

class InStateDataType:
    '''Class containig methods for working with inclusive state data.
    
    Reads important metadata from an inclusive state variable, and collects
    the counts of each inclusive state in a data set into a StateSum.
    '''
    bit_desc_re = re.compile('^bit_(\d+)_description$')
    def __init__(self, sumvar, ds):
        self.type_name = 'stateInclusive'
        self.flag_masks = []
        self.attrs = {a['name']:a['val'] for a in sumvar['attributes']}
        if 'flag_masks' in self.attrs:
            self.flag_masks = self.attrs['flag_masks']
        else:
            n_flags = 0
            for a,v in self.attrs.items():
                match = InStateDataType.bit_desc_re.match(a)
                if not match: continue
                n_flags = max(n_flags, int(match.group(1)))
            self.flag_masks = [2**x for x in range(n_flags)]
        self.flag_descriptions = ['']*len(self.flag_masks)
        for n in range(len(self.flag_masks)):
            bit_desc = 'bit_'+str(n+1)+'_description'
            if bit_desc in self.attrs:
                self.flag_descriptions[n] = self.attrs[bit_desc]
            elif 'flag_meanings' in self.attrs:
                self.flag_descriptions[n] = self.attrs['flag_meanings'].split()[n]

    @staticmethod
    def matches(sumvar, ds):
        return sumvar['dtype'][:3] == 'int' and (
            any(a['name'] == 'flag_masks' for a in sumvar['attributes']) or \
            any(InStateDataType.bit_desc_re.match(a['name']) for a in sumvar['attributes'])
            )

    def summarize(self, data=None):
        if not data:
            return StateSum([(str(m),0) for m in self.flag_masks])
        return StateSum([ (str(m), data[str(m)] if str(m) in data else 0) for m in self.flag_masks ])

    def columns(self):
        return ['bit '+str(n) for n in range(len(self.flag_masks))], self.flag_descriptions


class QCDataType(InStateDataType):
    '''Class containing methods for working with Quality control data

    Subclass of InStateData, only difference is that variable names must 
    start with a qc_ to be identified as QC and not just inclusive state.
    '''
    qc_bit_desc_re = re.compile('qc_bit_(\d+)_description')
    def __init__(self, sumvar, ds):
        InStateDataType.__init__(self, sumvar, ds)

        self.type_name = 'stateInclusiveQC'
        if not self.flag_masks or not self.flag_descriptions:
            n_flags = 0
            for attr in ds.attributes.keys():
                match = QCDataType.qc_bit_desc_re.match(attr)
                if not match: continue
                n_flags = max(n_flags, int(match.group(1)))
            self.flag_masks = [2**x for x in range(n_flags)]
            self.flag_descriptions = ['']*len(self.flag_masks)
            for n in range(len(self.flag_masks)):
                bit_desc = 'qc_bit_'+str(n+1)+'_description'
                if bit_desc in ds.attributes:
                    self.flag_descriptions[n] = ds.attributes[bit_desc][0].val

    @staticmethod
    def matches(sumvar, ds):
        if sumvar['name'].startswith('qc_'):
            if (InStateDataType.matches(sumvar, ds) or any(QCDataType.qc_bit_desc_re.match(a['name']) for a in ds.current_summary['attributes'])):
                return True
            logging.warning('Non-comformant QC: %s:%s' % (ds.current_summary['path'],sumvar['name']))
        return False

def data_type_of(sumvar, ds, typeonly=False):
    '''Determines the data type of ncvar
    '''
    for DataType in (DimlessDataType, QCDataType, ExStateDataType, InStateDataType, NumDataType):
        if DataType.matches(sumvar, ds):
            if typeonly:
                return DataType
            return DataType(sumvar, ds)

### Data classes ------------------------------------------------------------------------------------------------------

class TimedData:
    '''The TimedData object is used to summarize and write out all kinds of
    data with a time dimension.

    Attributes:
        ds         parent Datastream
        data       dictionary of sample time : summary object pairs
        data_type  provides interface to a specific type of data
    '''
    def __init__(self, sumvar, ds):
        self.ds = ds
        self.name = 'Data'
        self.data = { } # key is time in epoch, val is summary
        self._data_type = data_type_of(sumvar, ds, True)
        self.data_type = self._data_type(sumvar, ds)

    def load(self, sumvar):
        '''
        parameters:
        sumvar: variable summary dict
        '''
        self.var_name = sumvar['name']
        time = self.ds.current_summary['time']
        for i in range(len(time)):
            if self._data_type != data_type_of(sumvar, self.ds, True):
                raise Exception("Fatally inconsistent variable '%s' (at %s): %s and %s" % (
                    sumvar['name'],
                    utils.timetostr(time[i],'%Y-%m-%d'),
                    self._data_type,
                    data_type_of(sumvar, self.ds, True)
                ))
            s = self.data_type.summarize({ k:v[i] for k,v in sumvar['data'].items() })
            if time[i] not in self.data:
                self.data[time[i]] = self.data_type.summarize()
            self.data[time[i]] += s

    def get_nifs(self):
        #  adds up totals for both old a new, returns the sum as a tuple
        nmiss = 0
        nnans = 0
        ninfs = 0
        nfill = 0
        for ns in self.data.values():
            a = (0, 0, 0, 0) if ns is None else ns.get_nifs()
            nmiss += a[0] 
            nnans += a[1]
            ninfs += a[2]
            nfill += a[3]
        return (nmiss, nnans, ninfs, nfill)

    def jsonify(self):

        columns, tooltips = self.data_type.columns()

        if len(self.data) == 1:
            val = next(iter(self.data.values())).row()
            sec = {
                'type': 'staticSummary',
                'name': self.name,
                'columns': columns,
                'tooltips': tooltips,
                'val': val
            }
            if hasattr(self, '_difference'):
                sec['difference'] = self._difference
            return sec

        # Format data for a csv file...
        columns = ['beg', 'end']+columns
        tooltips = ['', '']+tooltips
        csv = [[time, time+self.ds.sample_interval]+summary.row() for time, summary in sorted(self.data.items())]
        csv = [columns, tooltips]+csv

        plot_json = {
                'type': 'plot',
                'data': csv,
                'ds_path': self.ds.path
            }

        if self.ds.use_dq_inspector:
            plot_json['var_name'] = self.var_name

        return utils.json_section(self, [plot_json])
            
class UntimedData(Timeline):
    '''Summarizes variable data which lacks a time dimension.
    Stores a file-by-file summary of the data in its Timeline superclass.

    Attributes:
        ds         parent Datastream
        data_type  provides interface to a specific type of data
    '''
    def __init__(self, sumvar, ds):
        Timeline.__init__(self, 'Data', ds)
        self.data_type = data_type_of(sumvar, ds)

    def load(self, sumvar):
        summ = self.data_type.summarize(sumvar['data'])
        Timeline.load(self, summ)

    def jsonify(self):

        columns, tooltips = self.data_type.columns()

        if len(self) == 1:
            sec = {
                'type': 'staticSummary',
                'name': self.name,
                'columns': columns,
                'tooltips': tooltips,
                'val': self[0].val.row()
            }
            if hasattr(self, '_difference'):
                sec['difference'] = self._difference
            return sec

        columns = ['beg', 'end']+columns
        tooltips = ['', '']+tooltips
        csv = [[self.ds.file_timeline[log.beg].beg, self.ds.file_timeline[log.end].end]+log.val.row() for log in self]
        csv = [columns, tooltips]+csv

        return utils.json_section(self, [
            {
                'type': 'plot',
                'data': csv,
                'separate': ['data']
           }
        ])


### Variable ----------------------------------------------------------------------------------------------------------

class Variable:
    '''Stores summary information about a variable in a datastream.

    Attributes:
        name        variable name
        dims        timeline where values are tuples of variable dimensions
        dtype       variable's data type (numpy name)
        attributes  timeline dict of the variable's attributes
        companions  QC and other companion variables get stored in this dict
    '''
    def __init__(self, sumvar, ds):
        self.ds = ds
        self.name = sumvar['name']
        self.dims = Timeline('Dimensions', ds)
        self.dtype = Timeline('Data Type', ds)
        self.attributes = TimelineDict('Attributes', ds)
        self.companions = VariableDict('Companions', ds)
        self.companion_names = set()
        self.metadata_only = 'data' not in sumvar
        if not self.metadata_only:
            self.data = TimedData(sumvar, ds) if 'time' in sumvar['dimensions'] else \
                        UntimedData(sumvar, ds)

    def load(self, sumvar):
        # load metadata
        self.dims.load(sumvar['dimensions'])
        self.dtype.load(sumvar['dtype'])
        self.attributes.load({a['name']:a['val'] for a in sumvar['attributes']})
        if 'companions' in sumvar:
            self.companion_names = self.companion_names | set(sumvar['companions'])
        if not self.metadata_only:
            self.data.load(sumvar)

    def get_nifs(self):
        if type(self.data) is TimedData:
            return self.data.get_nifs()
        return (0, 0, 0, 0)

    def jsonify(self):
        sec = utils.json_section(self, [
            self.dtype.jsonify(),
            self.dims.jsonify(),
            self.attributes.jsonify(),
        ])
        if not self.metadata_only:
            sec['contents'].append(self.data.jsonify())

        if self.companions:
            sec['contents'].append(self.companions.jsonify())

        sec['type'] = 'variable'
        sec['dims'] = self.dims[0].val if len(self.dims) == 1 else 'varying'

        return sec

### Dicts -------------------------------------------------------------------------------------------------------------

class NCDict(dict):
    def __init__(self, name, ds):
        dict.__init__(self)
        self.name = name
        self.ds   = ds
    
    def jsonify(self):
        ''''''
        return utils.json_section(self, [x.jsonify() for x in self.values()])

class TimelineDict(NCDict):
    '''Extension of the dictionary class specialized for loading in name: timeline pairs.
    '''
    def __init__(self, name, ds):
        NCDict.__init__(self, name, ds)

    def load(self, data):
        '''
        data: a dictionary of name: data pairs. Each name will be associated with its own timeline.
        '''
        for name, val in data.items():
            if name not in self:
                self[name] = Timeline(name, self.ds)
            self[name].load(val)

class VariableDict(NCDict):

    def load(self, data):
        '''
        data: a dictionary of name: ncvar pairs, where ncvar is a netCDF4 variable object
        '''
        for name, var in data.items():
            if name not in self:
                self[name] = Variable(var, self.ds)
            self[name].load(var)

    def _clear_companions(self):
        '''Remove companion variables from the top-level variable dict, so they don't exist twice.

        This is a recursive function which should only be called through nest_companions
        '''
        companion_names = set()
        for var in self.values():
            companion_names |= set(var.companions.keys())

        for var_name in companion_names:
            self.pop(var_name, None)

        for var in self.values():
            var.companions._clear_companions()

    def nest_companions(self):
        '''Moves companon variable such as qc_<var> down into the companions attr of their parent var
        '''
        for var in self.values():
            var.companions.update({n: v for n, v in self.items() if n in var.companion_names})

        self._clear_companions()

### Datastream --------------------------------------------------------------------------------------------------------

TimeInterval = namedtuple('TimeInterval', ['beg', 'end'])

class Datastream:
    '''Data structure storing summary information of a datastream.
    '''

    # BIG TODO: Create an alternate Datastream initializer method
    #           which reads in data from the mongodb.
    #           This could be WAY faster than reading the data in from files directly.
    def __init__(
            self,
            use_dq_inspector=True,
            sample_interval=None
            ):
        self.sample_interval  = sample_interval
        self.summary_times    = []
        self.file_timeline    = []
        self.attributes       = TimelineDict('Attributes', self)
        self.dimensions       = TimelineDict('Dimensions', self)
        self.variables        = VariableDict('Variables', self)
        self.ds_name          = None
        self.current_summary  = None
        self.use_dq_inspector = use_dq_inspector

    def add(self, summary):
        f = summary
        self.current_summary = f
        fn = os.path.basename(f['path'])
        if self.ds_name is None:
            self.path = os.path.dirname(f['path'])
            self.ds_name = utils.file_datastream(fn)
        else:
            ds_name = utils.file_datastream(fn)
            if ds_name != self.ds_name:
                raise RuntimeError(path+' contains files from different datastreams, '+ \
                    self.ds_name+' and '+ds_name+'.')

        beg, end = int(f['span'][0]), int(f['span'][1])

        self.summary_times = sorted(
            set(f['time']) | \
            set(self.summary_times)
            )

        # Add this begin, end pair to the file timeline...
        if self.file_timeline and self.file_timeline[-1].end > beg:
            logging.warning('%s overlaps with previous file.' % f['path'])
        self.file_timeline.append(TimeInterval(beg, end))

        # Load global metadata...
        self.attributes.load({a['name']:a['val'] for a in f['attributes']})
        self.dimensions.load({d['name']:d['length'] for d in f['dimensions']})
        self.variables.load({v['name']:v for v in f['variables']})

    def summarize(self):
        nmiss = 0
        nanns = 0 
        infs = 0   
        fills = 0 

        for key, value in self.variables.items():
            if type(value) is Variable:
                a, b, c, d = value.get_nifs()
                nmiss += a
                nanns += b
                infs  += c
                fills += d

        bad_data = {}
        bad_data['nmiss'] = nmiss
        bad_data['nanns'] = nanns
        bad_data['infs'] =  infs
        bad_data['fills'] = fills

        return {
            'type': 'summary',
            'bad_data': bad_data,
        }

    def jsonify(self):
        self.variables.nest_companions()
        return {
            'type': 'datastream',
            'ds_name': self.ds_name,
            'path': self.path,
            'sample_interval': self.sample_interval,
            'summary_times': self.summary_times,
            'contents': [
                { 
                    'type': 'section',
                    'name': 'File Timeline',
                    'contents': [
                        {
                            'type': 'fileTimeline',
                            'data': [['beg', 'end']]+self.file_timeline
                        }
                    ]
                },
                self.attributes.jsonify(),
                self.dimensions.jsonify(),
                self.variables.jsonify(),
                self.summarize(),
            ]
        }

    def json(self):
        return json.dumps(self.jsonify(), default=utils.JEncoder)
