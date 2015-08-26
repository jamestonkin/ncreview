'''Datastream comparison classes for the ncreview tool.

This module contains the classes nessecary to perform a comparison of two 
datastreams and output the resulting comparison report to a json file which 
can be rendered by the web tool.

Recurring Attributes

A name attribute is the name of the object's corresponding section in the
    web-based UI.

A dsd attribute refers back to the DatastreamDiff that contains the object.

Recurring methods:

Generally, a class's initializer generates the comparison data structure 
    from params old and new as the data structures to be compared.
    These old and new params' type generally indicated by the class name,
    for example DatastreamDiff compares two Datastreams,
    TimelineDiff compares two Timelines.
    the dsd parameter should take in the parent DatastreamDiff.

A difference() method returns 'same', 'changed', 'added', or 'removed'
    indicating the nature of that comparison object. These difference strings 
    are used later in the web tool to highlight entries accordingly.

A jsonify() method returns a data structure that can be converted to json
    and used to generate the report in the web-based UI.

'''

import os
import sys
import json
from collections import namedtuple 

from ncr.datastream import TimedData, UntimedData
import ncr.utils as utils
import pdb

### Timeline ----------------------------------------------------------------------------------------------------------

Diff = namedtuple('Diff', ['old', 'new', 'beg', 'end'])

class TimelineDiff(list):
    '''Comparison between two Timelines.
    Logs the differences between two timelines in a list of Diff objects.
    '''
    def __init__(self, name, old, new, dsd):
        super(TimelineDiff, self).__init__(self)
        self.name = name
        self.dsd = dsd

        for beg, end, old_i, new_i in utils.shared_times(dsd.old_file_times, dsd.new_file_times):
            old_val = next((l.val for l in old if l.beg <= old_i <= l.end), None)
            new_val = next((l.val for l in new if l.beg <= new_i <= l.end), None)

            if self and self[-1].old == old_val and self[-1].new == new_val:
                self[-1] = Diff(self[-1].old, self[-1].new, self[-1].beg, end)
            else:
                self.append(Diff(old_val, new_val, beg, end))

    @utils.store_difference
    def difference(self):
        if not self: 
            return 'same'

        diff = lambda d: \
             'same'    if d.old == d.new else \
             'added'   if d.old is None else \
             'removed' if d.new is None else \
             'changed'

        first = diff(self[0])

        if first == 'changed' or all(diff(d) == first for d in self):
            return first
        else:
            return 'changed'

    def jsonify(self):
        if len(self) == 1:
            sec = {
                'type': 'staticValueDiff',
                'name': self.name,
                'old': self[0].old,
                'new': self[0].new
            }
            if hasattr(self, '_difference'):
                sec['difference'] = self._difference
            return sec
        else:
            return utils.json_section(self, [
                {
                    'type': 'timelineDiff',
                    'data': [['old', 'new', 'beg', 'end']]+[[d.old, d.new, d.beg, d.end] for d in self]
                }
            ])

def compare_timelines(name, old, new, dsd):
    td = TimelineDiff(name, old, new, dsd)
    if td.difference() == 'same':
        setattr(new, '_difference', 'same')
        setattr(new, 'difference', lambda: 'same')
        return new
    return td

### Data --------------------------------------------------------------------------------------------------------------

# TODO: Create a new kind of object TimedDataDelta which will plot new minus old data.
# TODO: Yan wants a feature where differences between individual values are plotted
# TODO: somebody else wants a feature where a density of scatterpoints of old and new is plotted

class TimedDataDiff:
    '''Comparison of old and new timed data.
    '''
    def __init__(self, old, new, dsd):
        self.var_name = old.var_name
        self.dsd = dsd
        self.name = 'Data'
        self.data_type = new.data_type
        self.old = [old.data[t] if t in old.data else None for t in dsd.summary_times]
        self.new = [new.data[t] if t in new.data else None for t in dsd.summary_times]

    @utils.store_difference
    def difference(self):
        if not self.old and not self.new:
            return 'same'

        diff = lambda o, n: \
            'same'    if o == n else \
            'added'   if o is None else \
            'removed' if n is None else \
            'changed'
        
        summary_times = self.dsd.summary_times
        sample_interval = self.dsd.sample_interval

        shared_times = list(utils.shared_times(self.dsd.old_file_times, self.dsd.new_file_times))

        # get the first difference
        def sample_diffs():
            i = 0
            for beg, end, *_ in shared_times:
                beg = (beg//sample_interval)*sample_interval

                while i < len(summary_times) and summary_times[i] < beg: i += 1

                while i < len(summary_times) and summary_times[i] <= end:
                    yield diff(self.old[i], self.new[i])
                    i += 1

        sample_diffs = sample_diffs()

        first = next(sample_diffs, 'same')

        # if the first one is changed, we don't need to check any more
        if first == 'changed':
            return first

        # check the remaining differences
        for d in sample_diffs:
            if d != first:
                return 'changed'

        return first

    def jsonify(self):

        columns, tooltips = self.data_type.columns()

        if len(self.dsd.summary_times) == 1:
            sec = None
            if self.old[0] != self.new[0]:
                sec = {
                    'type': 'staticSummaryDiff',
                    'name': self.name,
                    'columns': columns,
                    'tooltips': tooltips,
                    'old' : self.old[0].row(),
                    'new' : self.new[0].row()
                }
            else:
                sec = {
                    'type': 'staticSummary',
                    'name': self.name,
                    'val' : self.new[0].row()
                }

            sec['difference'] = self.difference()
            return sec

        columns = ['beg', 'end']+columns;
        tooltips = ['','']+tooltips;
        old_csv = [columns, tooltips] + \
        [
            [t, t+self.dsd.sample_interval]+(x.row() if x is not None else []) \
            for t, x in zip(self.dsd.summary_times, self.old)
        ]
        new_csv = [columns, tooltips] + \
        [
            [t, t+self.dsd.sample_interval]+(x.row() if x is not None else [])  
            for t, x in zip(self.dsd.summary_times, self.new)
        ]
        
        # add nones to complete any empty rows
        for csv in old_csv, new_csv:
            length = max(map(len, csv))
            if length == 2: continue
            for row in csv:
                if len(row) == 2:
                    row += [None]*(length-2)

        plotDiff_json = {
                'type': 'plotDiff',
                'old_data': old_csv,
                'new_data': new_csv,
                'old_ds_path': self.dsd.old_path,
                'new_ds_path': self.dsd.new_path
           }

        if self.dsd.use_dq_inspector:
            plotDiff_json['var_name'] = self.var_name

        return utils.json_section(self, [plotDiff_json])

class UntimedDataDiff(TimelineDiff):
    '''Comparison of old and new untimed data.
    '''
    def __init__(self, old, new, dsd):
        TimelineDiff.__init__(self, 'Data', old, new, dsd)
        self.data_type = new.data_type
    
    def jsonify(self):
        columns, tooltips = self.data_type.columns()

        if len(self) == 1:
            sec = None
            if self[0].old != self[0].new:
                sec = {
                    'type': 'staticSummaryDiff',
                    'name': self.name,
                    'columns': columns,
                    'tooltips': tooltips,
                    'old' : self[0].old.row(),
                    'new' : self[0].new.row()
                }
            else:
                sec = {
                    'type': 'staticSummary',
                    'name': self.name,
                    'columns': columns,
                    'tooltips': tooltips,
                    'val' : self[0].new.row()
                }

            sec['difference'] = self.difference()
            return sec
        
        columns = ['beg', 'end']+columns
        tooltips = ['','']+tooltips;
        old_csv = [columns, tooltips]+[[d.beg, d.end]+(d.old.row() if d.old is not None else []) for d in self]
        new_csv = [columns, tooltips]+[[d.beg, d.end]+(d.new.row() if d.new is not None else []) for d in self]

        # add nones to complete any empty rows
        for csv in old_csv, new_csv:
            length = max(map(len, csv))
            if length == 2: continue
            for row in csv:
                if len(row) == 2:
                    row += [None]*(length-2)

        return utils.json_section(self, [
            {
                'type': 'plotDiff',
                'data_type': self.data_type.type_name,
                'old_data': old_csv,
                'new_data': new_csv
            }
        ])

def compare_data(old, new, dsd):
    '''Generic data comparison function.
    '''
    if type(old) != type(new) or type(old.data_type) != type(new.data_type):
        raise ValueError('cannot compare data summaries of different type')
    if isinstance(old, TimedData):
        return TimedDataDiff(old, new, dsd)

    if isinstance(old, UntimedData):
        return UntimedDataDiff(old, new, dsd)

### Variable ----------------------------------------------------------------------------------------------------------

class VariableDiff:
    '''Comparison of old and new variables.

    Attribtues:
        name        Variable name
        dims        TimelineDiff of variables' dimensions
        dtype       TimelineDiff of vareiables' data types
        attributes  TimelineDictDiff of the variables' attributes
        companions  VariableDictDiff of the variables' companion variables
        data        Comparison of the variables' data
        old_data    if the old and new data types are incomparable, this stores old data
        new_data    if the old and new data types are incomparable, this stores new data
    '''
    def __init__(self, name, old, new, dsd):
        self.name = name
        self.dims = compare_timelines('Dimensions', old.dims, new.dims, dsd)
        self.dtype = compare_timelines('Data Type', old.dtype, new.dtype, dsd)
        self.attributes = TimelineDictDiff('Attributes', old.attributes, new.attributes, dsd)
        self.companions = VariableDictDiff('Companions', old.companions, new.companions, dsd)
        self.data = None
        self.old_data = None
        self.new_data = None
        if not old.metadata_only or not new.metadata_only:
            try:
                self.data = compare_data(old.data, new.data, dsd)
            except ValueError as e:
                # create data to display error later
                dsd.has_warnings = True
                if not hasattr(dsd, 'incomparable_summaries'):
                    dsd.incomparable_summaries = []
                dsd.incomparable_summaries.append((name, old.data.data_type.type_name, new.data.data_type.type_name))

                self.old_data = old.data
                self.old_data.name = 'Old Data'
                self.new_data = new.data
                self.new_data.name = 'New Data'

    @utils.store_difference
    def difference(self):
        first = self.dims.difference()

        if first == 'changed' or not self.data:
            return 'changed'

        if first == self.dtype.difference() and \
           first == self.attributes.difference() and \
           first == self.companions.difference() and \
           first == self.data.difference():
           return first
        else:
            return 'changed'

    def jsonify(self):
        contents = [
            self.dtype.jsonify(),
            self.dims.jsonify(),
            self.attributes.jsonify(),
        ]
        
        if self.data:
            contents.append(self.data.jsonify())
        elif self.old_data and self.new_data:
            contents += self.old_data.jsonify(), self.new_data.jsonify()

        sec = utils.json_section(self, contents)

        if self.companions:
            sec['contents'].append(self.companions.jsonify())

        sec['type'] = 'variableDiff'
        if len(self.dims) == 1:
            if isinstance(self.dims, TimelineDiff):
                if self.dims[0].old == self.dims[0].new:
                    sec['dims'] = self.dims[0].new
                else:
                    sec['dims'] = 'varying'
            else:
                sec['dims'] = self.dims[0].val
        else: 
            sec['dims'] = 'varying'
        return sec

### Dicts -------------------------------------------------------------------------------------------------------------

class NCDictDiff(dict):
    '''Extention of the dictionary story nc objects, either attributes or variable summaries.
    '''
    def __init__(self, name, old, new, dsd, constructor):
        super(NCDictDiff, self).__init__(self)
        self.name = name    
        self.dsd = dsd  
        for name in set(old.keys())|set(new.keys()):
            if name in old and name in new:
                self[name] = constructor(name, old[name], new[name], dsd)
            elif name in old:
                self[name] = old[name]
                setattr(self[name], '_difference', 'removed')
            elif name in new:
                self[name] = new[name]
                setattr(self[name], '_difference', 'added')

    @utils.store_difference
    def difference(self):
        if not self: return 'same'

        get_difference = lambda x : \
            x.difference() if hasattr(x, 'difference') else \
            x._difference  if hasattr(x, '_difference') else \
            'same'

        first = get_difference(next(iter(self.values())))
    
        if all(get_difference(d) == first for d in self.values()):
            return first
        
        return 'changed'

    def jsonify(self):

        n_diffs = {
            'same'    : 0,
            'changed' : 0,
            'added'   : 0,
            'removed' : 0
        }           

        for val in self.values():
            diff = val.difference() if hasattr(val, 'difference') else \
                   val._difference  if hasattr(val, '_difference') else \
                   'same'

            n_diffs[diff] += 1

        sec = utils.json_section(self, [t.jsonify() for t in self.values()])
        sec['type'] = 'groupDiff'
        sec['n_diffs'] = n_diffs
        return sec


class TimelineDictDiff(NCDictDiff):
    def __init__(self, name, old, new, dsd):
        NCDictDiff.__init__(self, name, old, new, dsd, compare_timelines)

class VariableDictDiff(NCDictDiff):
    def __init__(self, name, old, new, dsd):
        NCDictDiff.__init__(self, name, old, new, dsd, VariableDiff)
        
### Datastream --------------------------------------------------------------------------------------------------------

def ftime_difference(old_ftimes, new_ftimes):
    '''Compare two file timelines and return their difference.

    This is separated out into a function purely to keep DatastreamDiff's jsonify() looking tidy.
    '''
    if old_ftimes and not new_ftimes:
        return 'removed'
    elif new_ftimes and not old_ftimes:
        return 'added'
    elif all(a == b for a, b in zip(old_ftimes, new_ftimes)):
        return 'same'
    else:
        return 'changed'

class DatastreamDiff:
    def __init__(self, old, new):
        self.sample_interval = old.sample_interval

        if old.sample_interval != new.sample_interval:
            raise ValueError('Old and new datastreams must share the same sample interval')
        self.has_warnings = False
        self.old_path = old.path
        self.new_path = new.path
        self.old_ds_name = old.ds_name
        self.new_ds_name = new.ds_name
        self.summary_times = sorted(set(old.summary_times)|set(new.summary_times))
        self.old_file_times = old.file_timeline
        self.new_file_times = new.file_timeline
        self.attributes = TimelineDictDiff('Attributes', old.attributes, new.attributes, self)
        self.dimensions = TimelineDictDiff('Dimensions', old.dimensions, new.dimensions, self)
        self.variables = VariableDictDiff('Variables', old.variables, new.variables, self)
        self.use_dq_inspector = old.use_dq_inspector and new.use_dq_inspector

        if self.has_warnings:
            if hasattr(self, 'incomparable_summaries'):
                sys.stderr.write('\n%d variable summaries are of different type and cannot be compared:\n'% \
                    len(self.incomparable_summaries)
                    )
                sys.stderr.write('\n'.join('%s - Old: %s, New: %s'%x for x in self.incomparable_summaries))
                sys.stderr.write('\n')
            sys.stderr.flush()

    def jsonify(self):
        return {
            'type': 'datastreamDiff',
            'old_path': self.old_path,
            'new_path': self.new_path,
            'old_ds_name': self.old_ds_name,
            'new_ds_name': self.new_ds_name,
            'sample_interval': self.sample_interval,
            'summary_times': self.summary_times,
            'contents': [
                {  
                    'type': 'section',
                    'name': 'File Timeline',
                    'difference': ftime_difference(self.old_file_times, self.new_file_times),
                    'contents': [
                        {
                            'type': 'fileTimelineDiff',
                            'old_data': [['beg', 'end']]+self.old_file_times,
                            'new_data': [['beg', 'end']]+self.new_file_times
                        }
                    ]
                },
                self.attributes.jsonify(),
                self.dimensions.jsonify(),
                self.variables.jsonify()
            ]
        }

    def json(self):
        j = self.jsonify()
        return json.dumps(j, default=utils.JEncoder)
