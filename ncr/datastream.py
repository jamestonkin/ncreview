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
import numpy as np
import datetime as dt
from netCDF4 import Dataset
from collections import namedtuple

import ncr.utils as utils

import pdb

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
        return isinstance(other, DimlessSum) and\
               (self.val == other.val or (np.isnan(self.val) and np.isnan(other.val)))

    def __ne__(self, other):
        return not self.__eq__(other)

    def row(self):
        try:
            if (hasattr(self.val, 'mask') and bool(self.val.mask)) or np.isnan(self.val):
                return [None]
        except:
            pass
        return [self.val]

class NumSum:
    '''Summary of the values in a variable holding numerical data.

    This numerical summary (NumSum) holds summary statistics of a dataset. the +=
    operator correctly combines two statistical summaries, such that for
    datasets a and b, numsum(a)+numsum(b) = numsum(a concatenated with b).
    '''
    __slots__ = \
        ('n', 'ngood', 'nmiss', 'ninf', 'nnan', 'nfill', 'min', 'max', 'mean', 'var')

    def __init__(self,
            n     = 0,
            ngood = 0,
            nmiss = 0,
            ninf  = 0,
            nnan  = 0,
            nfill = 0,
            mn    = float('inf'),
            mx    = -float('inf'),
            mean  = float('nan'),
            var   = float('nan')
            ):

        self.n     = n
        self.ngood = ngood
        self.nmiss = nmiss
        self.ninf  = ninf
        self.nnan  = nnan
        self.nfill = nfill
        self.min   = mn 
        self.max   = mx
        self.mean  = mean
        self.var   = var

    def __iadd__(self, other):
        s, o = self, other # for brevity
        ngood = s.ngood + o.ngood
        try:
            mean = s.mean if np.isnan(o.mean) else \
                   o.mean if np.isnan(s.mean) else \
                   (s.mean*s.ngood + o.mean*o.ngood)/ngood \
                   if ngood > 0 else float('nan')
        except TypeError:
            mean = float('nan')

        try:
            if np.isnan(o.var): pass
            elif np.isnan(s.var):
                s.var = o.var
            elif ngood > 0:
                # http://stats.stackexchange.com/questions/43159/ \
                # how-to-calculate-pooled-variance-of-two-groups-given-known-group-variances-mean
                # NOTE: I added the abs() because you can never end up with a negative variance and apparently
                # sometimes if the number is really close to zero you might end up with it looking negative.
                s.var = abs( (s.ngood * (s.var+s.mean**2) + o.ngood * (o.var+o.mean**2)) / ngood - mean**2 )
            else:
                s.var = float('nan')

        except TypeError:
            s.var = float('nan')

        s.n     = s.n + o.n
        s.ngood = ngood
        s.nmiss = s.nmiss + o.nmiss
        s.nnan  = s.nnan  + o.nnan
        s.ninf  = s.ninf  + o.ninf
        s.nfill = s.nfill + o.nfill
        s.min   = min(s.min, o.min)
        s.max   = max(s.max, o.max)
        s.mean  = mean

        return s
    
    def __eq__(self, other):
        '''Tests the equivalence of two numerical summaries to within plausible rounding errors'''
        if not isinstance(other, NumSum):
            return False
        for att in NumSum.__slots__:
            s = getattr(self, att)
            o = getattr(other, att)
            if np.isnan(s) and np.isnan(o):
                continue
            if np.isinf(s) and np.isinf(o) and s == o:
                continue
    
            d = np.abs(s - o)
            if d > 1e-4 or np.isnan(d):
                return False

        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def row(self):
        '''Converts the summary into a list for storage in a table as a row'''
        s = self
        return [
            s.n, s.ngood, s.nmiss, s.ninf, s.nnan, s.nfill, 
            s.min if not np.isinf(s.min) and not np.isnan(s.min) else None, 
            s.max if not np.isinf(s.max) and not np.isnan(s.max) else None, 
            s.mean if not np.isinf(s.mean) and not np.isnan(s.mean) else None,
            np.sqrt(s.var) if not np.isinf(s.var) and not np.isnan(s.var) and s.var >=0 else None]

    def jsonify(self):
        data = {
            'n'    : self.n,
            'ngood': self.ngood,
            'nmiss': self.nmiss,
            'ninf' : self.ninf,
            'nnan' : self.nnan,
            'min'  : self.min if not np.isinf(self.min) and not np.isnan(self.min) else None,
            'max'  : self.max if not np.isinf(self.max) and not np.isnan(self.max) else None,
            'mean' : self.mean if not np.isinf(self.mean) and not np.isnan(self.mean) else None,
            'std'  : np.sqrt(self.var) if not np.isinf(self.var) and not np.isnan(self.var) else None
        }
        return data

class StateSum(list):
    '''Summary of a state indicator variable.

    Holds an array with the number of occurrences of each state in a state 
    indicator variable. the += operator combines two state summaries by the 
    second operand's flag counts to the first.
    '''
    def __init__(self, vals):
        if vals and isinstance(vals[0], list): print(vals)
        list.__init__(self)
        self[:] = list(map(int, vals))

    def __iadd__(self, other):
        if len(self) != len(other):
            raise ValueError('two state summaries must be of the same length to combine.')

        for i, x in enumerate(other):
            self[i] += x

        return self

    def __eq__(self, other):
        return isinstance(other, StateSum) and\
               len(self) == len(other)     and\
               all(a == b for a,b in zip(self, other))

    def __ne__(self, other):
        return not self.__eq__(other)

    def row(self):
        return self[:]

### Data Types --------------------------------------------------------------------------------------------------------

class DimlessDataType:
    '''Wraps a single value with the expected DataType interface.
    '''
    def __init__(self, ncvar, ds):
        self.type_name = 'dimless'

    @staticmethod
    def matches(ncvar, ds):
        return not ncvar.dimensions

    def summarize(self, data=None):
        return DimlessSum(data[0]) if data is not None else DimlessSum(None)

    def columns(self):
        return ['value'], ['value']


class NumDataType:
    '''Class containing methods for working with numerical data

    Performs numerical summary of data and returns the result as a NumSum
    '''
    def __init__(self, ncvar, ds):
        self.type_name = 'numeric'

    @staticmethod
    def matches(ncvar, ds):
        return True

    def summarize(self, data=None, missing_value=-9999):
        '''Return summary statistics of the data as a NumSum object

        parameters:
        data: a 1-dimensional numpy array of numerical data to summarize
        missing_value: The value used for missing value in this dataset.
                       Defaults to -9999
        '''
        if data is None:
            return NumSum()

        size = int(data.size)

        nmiss, nfill = 0, 0
        if hasattr(data, 'mask'):
            masked = data[data.mask].data
            nmiss = int(np.sum(masked == missing_value))
            nfill = masked.size - nmiss

        try:
            nans = np.where(np.isnan(data))
            nnan = nans[0].size
        except TypeError:
            nnan = 0

        try:
            infs = np.where(np.isinf(data))
            ninf = infs[0].size
        except TypeError:
            ninf = 0

        if nnan or ninf:
            if not hasattr(data, 'mask'):
                data = np.ma.MaskedArray(data, mask = False)
            if nnan:
                data.mask[nans] = True
            if ninf:
                data.mask[infs] = True

        if hasattr(data, 'mask'):
            data = data.compressed()

        numsum = NumSum(n=size, ngood=data.size, nmiss=nmiss, nnan=nnan, ninf=ninf, nfill=nfill)

        try:
            if data.size:
                if data.dtype == np.dtype('S1'):
                    data = data.astype(int)
                numsum.min   = float(data.min())
                numsum.max   = float(data.max())
                numsum.mean  = data.mean(dtype=np.float64)
                numsum.var   = data.var(dtype=np.float64)

        except:
            pass
        
        return numsum

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
            'Standard deviation']

        return cols, tooltips


class ExStateDataType:
    '''Class containig methods for working with exclusive state data.

    Reads important metadata from an exclusive state variable, and collects
    the counts of each exclusive state in a data set into a StateSum.
    '''
    flag_desc_re = re.compile('^flag_(\d+)_description$')
    def __init__(self, ncvar, ds):
        '''Initialize an exclusive state variable

        Scans a netCDF4 variable object meeting exclusive state variable requirements
        to collect a list of valid values in that exclusive state in self.flag_values
        and the descriptions of those values in self.flag_descriptions
        '''
        self.type_name = 'exclusiveState'
        # get flag values
        self.flag_values = []
        self.var_name = ncvar._name
        
        if hasattr(ncvar, 'flag_values'):
            self.flag_values = list(ncvar.flag_values)
        else:
            n_flags = 0
            for attr in ncvar.ncattrs():
                match = ExStateDataType.flag_desc_re.match(attr)
                if not match: continue
                n_flags = max(n_flags, int(match.group(1)))

            self.flag_values = range(1, n_flags+1)

        # get flag descriptions
        self.flag_descriptions = ['']*len(self.flag_values)

        for n in range(len(self.flag_values)):
            # allow for gaps in numbers
            flag_num = self.flag_values[n]
            flag_desc = 'flag_'+str(flag_num+1)+'_description'
            if hasattr(ncvar, flag_desc):
                self.flag_descriptions[n] = getattr(ncvar, flag_desc)
            elif hasattr(ncvar, 'flag_meanings'):
                self.flag_descriptions[n] = ncvar.flag_meanings.split()[n]

    def columns(self):
        return ['flag '+str(v) for v in self.flag_values], self.flag_descriptions

    @staticmethod
    def matches(ncvar, ds):
        '''Checks if ncvar is an exclusive state variable

        Requirements:
            data is of integral type and one of the following:
            - has attribute 'flag values'
            - has one or more attributes with names matching the expected regex for a flag description

        Parameters
        ncvar: netcdf4 variable to check
        ds: datastream the netcdf4 variable is part of
        '''
        return (hasattr(ncvar, 'flag_values') or \
            any(ExStateDataType.flag_desc_re.match(a) for a in ncvar.ncattrs())) and \
            issubclass(ncvar.dtype.type, np.integer)

    def summarize(self, data=None):
        '''Count the ocurrences of each value in flag_values.
        
        Returns counts as a StateSum list parallel to flag_values.
        '''
        if data is None:
            return StateSum([0]*len(self.flag_values))

        d = data.astype(int)
        if hasattr(d, 'mask'):
            d = d.compressed()

        # bincount will fail on values below zero, so subtracting the minimum value
        # shifts the smallest value in array, negative or positive, to zero.
        mn = d.min()
        counts = np.bincount(d-mn)
        return  StateSum([counts[v-mn] if 0 <= v-mn < len(counts) else 0 for v in self.flag_values])


class InStateDataType:
    '''Class containig methods for working with inclusive state data.
    
    Reads important metadata from an inclusive state variable, and collects
    the counts of each inclusive state in a data set into a StateSum.
    '''
    bit_desc_re = re.compile('^bit_(\d+)_description$')
    def __init__(self, ncvar, ds):
        self.type_name = 'inclusiveState'

        # get flag masks
        self.flag_masks = []

        if hasattr(ncvar, 'flag_masks'):
            self.flag_masks = list(ncvar.flag_masks)
        else:
            n_flags = 0
            for attr in ncvar.ncattrs():
                match = InStateDataType.bit_desc_re.match(attr)
                if not match: continue
                n_flags = max(n_flags, int(match.group(1)))

            self.flag_masks = [2**x for x in range(n_flags)]

        self.flag_descriptions = ['']*len(self.flag_masks)

        # get flag descriptions
        for n in range(len(self.flag_masks)):
            bit_desc = 'bit_'+str(n+1)+'_description'
            if hasattr(ncvar, bit_desc):
                self.flag_descriptions[n] = getattr(ncvar, bit_desc)
            elif hasattr(ncvar, 'flag_meanings'):
                self.flag_descriptions[n] = ncvar.flag_meanings.split()[n]

    @staticmethod
    def matches(ncvar, ds):
        return (hasattr(ncvar, 'flag_masks') or any(InStateDataType.bit_desc_re.match(a) for a in ncvar.ncattrs()))\
            and issubclass(ncvar.dtype.type, np.integer)

    def summarize(self, data=None):
        if data is None:
            return StateSum([0]*len(self.flag_masks))

        d = data.astype(int)
        if hasattr(d, 'mask'):
            d = d.compressed()

        return StateSum([np.sum(d&m > 0) for m in self.flag_masks])

    def columns(self):
        return ['bit '+str(n) for n in range(len(self.flag_masks))], self.flag_descriptions


class QCDataType(InStateDataType):
    '''Class containing methods for working with Quality control data

    Subclass of InStateData, only difference is that variable names must 
    start with a qc_ to be identified as QC and not just inclusive state.
    '''
    qc_bit_desc_re = re.compile('qc_bit_(\d+)_description')
    def __init__(self, ncvar, ds):
        InStateDataType.__init__(self, ncvar, ds)

        self.type_name = 'qualityControl'
        if not self.flag_masks or not self.flag_descriptions:
            n_flags = 0
            for attr in ds.attributes.keys():
                match = QCDataType.qc_bit_desc_re.match(attr)
                if not match: continue
                n_flags = max(n_flags, int(match.group(1)))

            self.flag_masks = [2**x for x in range(n_flags)]

            # get flag descriptions
            self.flag_descriptions = ['']*len(self.flag_masks)
            for n in range(len(self.flag_masks)):
                bit_desc = 'qc_bit_'+str(n+1)+'_description'
                if bit_desc in ds.attributes:
                    self.flag_descriptions[n] = ds.attributes[bit_desc][0].val

    @staticmethod
    def matches(ncvar, ds):
        if ncvar._name.startswith('qc'):
            if (InStateDataType.matches(ncvar, ds) or \
                    any(QCDataType.qc_bit_desc_re.match(a) for a in ds.attributes.keys())):

                return True
            else:
                ds.has_warnings = True
                if not hasattr(ds, 'non_conformant_qc'):
                    ds.non_conformant_qc = set()
                ds.non_conformant_qc.add(ncvar._name)
                return False
        return ncvar._name.startswith('qc_') and \
            (InStateDataType.matches(ncvar, ds) or \
             any(QCDataType.qc_bit_desc_re.match(a) for a in ds.attributes.keys())
             )

def data_type_of(ncvar, ds):
    '''Determines the data type of ncvar
    '''
    for DataType in (DimlessDataType, QCDataType, ExStateDataType, InStateDataType, NumDataType):
        if DataType.matches(ncvar, ds):
            return DataType(ncvar, ds)

### Data classes ------------------------------------------------------------------------------------------------------

class TimedData:
    '''The TimedData object is used to summarize and write out all kinds of
    data with a time dimension.

    Attributes:
        ds         parent Datastream
        data       dictionary of sample time : summary object pairs
        data_type  provides interface to a specific type of data
    '''
    def __init__(self, ncvar, ds):
        self.ds = ds
        self.name = 'Data'
        self.data = {} # key is time in epoch, val is summary
        self.data_type = data_type_of(ncvar, ds)

    def load(self, ncvar):
        '''
        parameters:
        ncvar: a netCDF4 variable object
        '''

        ## Reformat the data, select desired sample interval, and flatten
        self.var_name = ncvar._name
        var_data = None
        try:
            var_data = ncvar[:]
        except Exception as e:
            self.ds.has_warnings = True
            if not hasattr(self.ds, 'unreadable_data'):
                self.ds.unreadable_data = {}
            self.ds.unreadable_data[ncvar._name] = str(e)
            return

        # swap the time dimension into the first axis of the array
        if 'time' not in ncvar.dimensions:
            self.ds.has_warnings = True
            if not hasattr(self.ds, 'missing_time'):
                self.ds.missing_time = []
            self.ds.missing_time.append((ncvar._name, self.ds.file_timeline[-1].beg))
            return

        time_i = ncvar.dimensions.index('time')
        if time_i > 0:
            var_data = var_data.swapaxes(0, time_i)
        
        summary_times = (self.ds.sample_times // self.ds.sample_interval).astype(int)*self.ds.sample_interval

        for t in map(int, np.unique(summary_times)):
            if t not in self.data:
                self.data[t] = self.data_type.summarize()

            # select only the chunk at the desired time
            sample_data = var_data[summary_times == t] if summary_times[0] != summary_times[-1] else var_data
            
            # flatten the array
            sample_data = sample_data.ravel()

            ## summarize the data and update the summary
            self.data[t] += self.data_type.summarize(sample_data)

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

        # format data for a csv file
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
    def __init__(self, ncvar, ds):
        Timeline.__init__(self, 'Data', ds)
        self.data_type = data_type_of(ncvar, ds)

    def load(self, ncvar):
        var_data = None
        try:
            var_data = ncvar[:]
        except Exception as e:
            self.ds.has_warnings = True
            if not hasattr(self.ds, 'unreadable_data'):
                self.ds.unreadable_data = {}
            self.ds.unreadable_data[ncvar._name] = str(e)
            return

        var_data = var_data.ravel()

        summ = self.data_type.summarize(var_data)
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
    def __init__(self, ncvar, ds, metadata_only=False):
        self.ds = ds
        self.name = ncvar._name
        self.dims = Timeline('Dimensions', ds)
        self.dtype = Timeline('Data Type', ds)
        self.attributes = TimelineDict('Attributes', ds)
        self.companions = VariableDict('Companions', ds)
        if not metadata_only:
            self.data = TimedData(ncvar, ds) if 'time' in ncvar.dimensions else \
                        UntimedData(ncvar, ds)
        self.metadata_only = metadata_only

    def load(self, ncvar):
        # load metadata
        self.dims.load(list(map(str, ncvar.dimensions)))
        self.dtype.load(str(ncvar.dtype))
        self.attributes.load({a: list(v) if isinstance(v, np.ndarray) else v for a, v in ncvar.__dict__.items()})
        if not self.metadata_only:
            self.data.load(ncvar)

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
    '''Extension of the dictionary class specialized for loading in name: variable pairs.
    '''
    def __init__(self, name, ds, metadata_only=False):
        NCDict.__init__(self, name, ds)
        self.metadata_only = metadata_only

    def load(self, data):
        '''
        data: a dictionary of name: ncvar pairs, where ncvar is a netCDF4 variable object
        '''
        for name, var in data.items():
            if name not in self:
                self[name] = Variable(var, self.ds, self.metadata_only)
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
        companion_prefixes = {
          'fgp'  : 'fraction of good points',
          'be'   : 'best estimate',
          'qc'   : 'quality control',
          'aqc'  : 'ancillary quality control'
        }

        for var in self.values():
            var.companions.update({n: v for n, v in self.items() \
                if any(p+'_'+var.name == n for p in companion_prefixes)})

        self._clear_companions()

### Datastream --------------------------------------------------------------------------------------------------------

TimeInterval = namedtuple('TimeInterval', ['beg', 'end'])

_total_time = 0


class Datastream:
    '''Data structure storing summary information of a datastream.

    Attributes:
        path             path to the datastream directory
        sample_interval  length in seconds of interval over which time-series data is summarized
        summary_times    list of times at which data was summarized
        file_timeline    list of TimeInterval objects indicating the start and end dates of each file
        attributes       dict of attribute name : Attribute pairs
        dimensions       dict of dimension name : Dimension pairs
        variables        dict of variable name: Variable pairs
        time             array of sample times found in the files
    '''
    # BIG TODO: Create an alternate Datastream initializer method
    #           which reads in data from the mongodb.
    #           This could be WAY faster than reading the data in from files directly.
    #           This will probably fall to you, Daniel.
    #           All of this is yours now. Good luck.
    def __init__(
            self,
            path,
            beg=dt.datetime.min,
            end=dt.datetime.max,
            sample_interval=None,
            metadata_only=False,
            progress_bar=None
            ):
        '''
        Parameters:
        path    path to the datastream directory
        beg     datetime object specifying beginning limit on which to read in data
        beg     datetime object specifying ending limit on which to read in data
        sample_interval     length in seconds of interval over which time-series data is summarized
        metadata_only       Only read in netcdf metadata, ignore actual data (much faster)
        progress_bar        progress bar object (see ncreview.py)

        '''

        if sample_interval is None: # if sample interval isn't specified by user
            # automatically set the sample interval
            if end - beg > dt.timedelta(days=10): # if interval is more than 10 days
                sample_interval = 24*60*60 # set interval to 24 hr
            else:
                sample_interval = 60*60 # set interval to 1 hr
        elif sample_interval <= 0: # if user specified non-positive sample interval
            raise ValueError('Sample_interval must be a positive number, not ' + str(sample_interval))

        self.path            = path
        self.sample_interval = sample_interval
        self.summary_times   = []
        self.file_timeline   = []
        self.attributes      = TimelineDict('Attributes', self)
        self.dimensions      = TimelineDict('Dimensions', self)
        self.variables       = VariableDict('Variables', self, metadata_only)
        self.sample_times    = None
        self.ds_name         = None
        self.has_warnings    = False

        # check to see if path is dq_inspector compatible
        self.use_dq_inspector = True
        path_match = re.search('\/([a-z]{3})\/\\1[a-zA-Z0-9\.]+\s*$', path)
        if path_match is None:
            self.has_warnings = True
            self.use_dq_inspector = False
        
        def is_valid(fname):
            t = utils.file_time(fname)
            return t is not None and beg <= t <= end

        files = sorted(filter(is_valid, os.listdir(path)))

        if not files:
            raise RuntimeError(path+' contains no netCDF files in the specified time period.')

        self.ds_name = utils.file_datastream(files[0])

        for f in files:
            # make sure it has the same datastream name
            ds_name = utils.file_datastream(f)
            if ds_name != self.ds_name:
                raise RuntimeError(path+' contains files from different datastreams, '+ \
                    self.ds_name+' and '+ds_name+'.')

            ncfile = None
            try:
                # open the file
                ncfile = Dataset(path+'/'+f, 'r')

                # get the time in the file name in epoch
                ftime = time.mktime(utils.file_time(f).timetuple())

                # get the array of sample times in epoch
                sample_times = np.array([])
                beg, end = None, None
                ncvars = ncfile.variables
                if not metadata_only:
                    if 'time_offset' in ncvars and 'base_time' in ncvars:
                        sample_times = ncvars['time_offset'][:] + ncvars['base_time'][:]

                    elif 'time' in ncvars:
                        base_time = ncvars['base_time'][:] if 'base_time' in ncvars else ftime
                        sample_times = ncvars['time'][:] + (60*60*24)*base_time//(60*60*24)

                    # get begin and end dates of the file
                    beg = sample_times[ 0] if getattr(sample_times, 'size', 0) else ftime
                    end = sample_times[-1] if getattr(sample_times, 'size', 0) else ftime
                
                else:
                    base_time = ncvars['base_time'][0] if 'base_time' in ncvars else ftime
                    if 'time_offset' in ncvars:
                        beg = ncvars['time_offset'][0]  + base_time
                        try:
                            end = ncvars['time_offset'][-1] + base_time
                        except IndexError:
                            end = beg

                    elif 'time' in ncvars:
                        midnight = (60*60*24)*base_time//(60*60*24)
                        beg = ncvars['time'][0]  + midnight
                        end = ncvars['time'][-1] + midnight

                    else:
                        beg = base_time
                        end = base_time

                beg, end = int(beg), int(end)

                # get the new summary times
                self.summary_times = sorted(
                    set(sample_interval*np.unique(sample_times//sample_interval)) | \
                    set(self.summary_times))
                
                self.sample_times = sample_times

                # add this begin, end pair to the file timeline
                if self.file_timeline and self.file_timeline[-1].end > beg:
                    # TODO: turn this into a post-run warning
                    sys.stderr.write('Warning: '+path+'/'+f+' overlaps with previous file.\n')
                self.file_timeline.append(TimeInterval(beg, end))

                # load global metadata
                attr_dict = {a: list(v) if isinstance(v, np.ndarray) else v for a, v in ncfile.__dict__.items()}
                self.attributes.load(attr_dict)
                self.dimensions.load({n: len(d) for n,d in ncfile.dimensions.items()})
                self.variables.load(ncvars)

            except Exception as e:
                self.has_warnings = True
                if not hasattr(self, 'unreadable_files'):
                    self.unreadable_files = []
                self.unreadable_files.append((f, e))
            finally:
                if ncfile: ncfile.close()

            if progress_bar: progress_bar.update(os.stat(path+'/'+f).st_size)

        self.variables.nest_companions()
        self.summary_times = sorted(self.summary_times)

        # TODO: should probably come up with a better warning system...
        #       ...lol nahh
        if self.has_warnings:
            sys.stderr.write('\nWarnings for %s:\n'%path)
            if hasattr(self, 'unreadable_data'):
                sys.stderr.write('%d variables contain unreadable data:\n'%len(self.unreadable_data) + \
                    '\n'.join('%s: %s'%(var, error) for var, error in self.unreadable_data.items())+'\n')
            if hasattr(self, 'non_conformant_qc'):
                sys.stderr.write(
                    "%d QC variables are missing qc bit description attributes:\n"%len(self.non_conformant_qc) + \
                    '\n'.join(self.non_conformant_qc)+'\n')
            if hasattr(self, 'missing_time'):
                sys.stderr.write(
                    "%d instances of variables missing a time dimension they had in other files:\n"% \
                    len(self.missing_time) + \
                    '\n'.join(
                        '%s on %s'%(n, dt.datetime.utcfromtimestamp(t).isoformat()) for n, t in self.missing_time
                        )+'\n'
                    )
            if hasattr(self, 'unreadable_files'):
                sys.stderr.write(
                    "%d files could not be read from\n%s\n"%(len(self.unreadable_files), path) + \
                    '\n'.join('%s: %s'%(f, str(e)) for f, e in self.unreadable_files)+'\n'
                    )

            if self.use_dq_inspector == False:
                sys.stderr.write('\nWarning: \n' + path + \
                    '\nis not in correct directory structure to create dq_inspector plots\n')


        sys.stderr.flush()

    def jsonify(self):
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
                self.variables.jsonify()
            ]
        }

    def json(self):
        return json.dumps(self.jsonify(), default=utils.JEncoder)
