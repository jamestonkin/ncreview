import os
import re
import sys
import time
import logging
import traceback
import numpy as np
import netCDF4 as nc
from netCDF4 import Dataset
from datetime import datetime

### Data Types --------------------------------------------------------------------------------------------------------

# class TimeDataType:
#     '''
#     '''
#     def __init__(self, ncvar, attributes):
#         self.type_name = 'time'

#     @staticmethod
#     def matches(ncvar, attributes):
#         return ncvar._name in ('time', 'time_offset')

#     def summarize(self, data=None):
#         summary = {'n': data.size}
#         if data.size:
#             summary.update({
#                 'min': data.min().item(),
#                 'max': data.max().item()
#             })
#         return summary

class DimlessDataType:
    '''Wraps a single value with the expected DataType interface.
    '''
    def __init__(self, ncvar, attributes):
        self.type_name = 'dimless'

    @staticmethod
    def matches(ncvar, attributes):
        return not ncvar.dimensions

    def summarize(self, data=None):
        if data is not None:
            return data[0].item()
        else:
            return None

class NumDataType:
    '''Class containing methods for working with numerical data

    Performs numerical summary of data and returns the result as a NumSum
    '''
    def __init__(self, ncvar, attributes):
        self.type_name = 'numeric'
        self.var_name = ncvar._name
        self.ncvar = ncvar
        self.missing_value = ncvar.getncattr('missing_value') if 'missing_value' in ncvar.__dict__ else None
        if self.missing_value is not None:
            if np.dtype(type(self.missing_value)) != ncvar.dtype:
                ncvar.set_auto_mask(False) # Can't auto apply missing_value if type cannot be coerced
                try:
                    self.missing_value = np.array(self.missing_value).astype(ncvar.dtype)
                except:
                    self.missing_value = None
        if self.missing_value is None:
            self.missing_value = -9999
        self.fill_value = ncvar.getncattr('_FillValue') if '_FillValue' in ncvar.__dict__ else None
        if self.fill_value is None:
            for t,v in nc.default_fillvals.items():
                if np.dtype(t) == ncvar.dtype:
                    self.fill_value = v
                    break

    @staticmethod
    def matches(ncvar, attributes):
        return True

    def summarize(self, data=None):
        '''Return summary statistics of the data as a NumSum object
        '''
        # DIFFICULT POTENTIAL TODO: 
        # This function traverses the array many times, when
        # it should only need to traverse it twice:
        # once to get all the value counts and min, max, mean
        # and another to get standard deviation.
        # Some C code here could greatly outperform this.

        if data is None:
            return {
                'n'      : 0,
                'ngood'  : 0,
                'nmiss'  : 0,
                'nnan'   : 0,
                'ninf'   : 0,
                'nfill'  : 0,
                'min'    : None,
                'max'    : None,
                'mean'   : None,
                'median' : None,
                'var'    : None,
                'std'    : None,
            }

        size = int(data.size)

        nmiss, nfill = 0, 0
        if not hasattr(data, 'mask'):
            if self.ncvar.mask: # Mask auto applied and no missing/fill found.
                data = np.ma.MaskedArray(data, mask=False)
            else: # Need to apply missing/fill mask manually.
                data = np.ma.masked_where(
                    (data == self.missing_value) |\
                    (data == self.fill_value), data, copy=False)
        masked = data[data.mask].data
            
        nfill = int(np.size(masked))
        if self.missing_value is not None:
            nmiss = int(np.sum(masked == self.missing_value))
        if self.fill_value is not None:
            nfill = int(np.sum(masked == self.fill_value))
        elif nfill > 0:
            nfill -= nmiss

        try:
            nans = np.where(np.isnan(data))
            nnan = int(nans[0].size)
        except TypeError:
            nnan = 0

        try:
            infs = np.where(np.isinf(data))
            ninf = int(infs[0].size)
        except TypeError:
            ninf = 0

        if nnan or ninf:
            if nnan:
                data.mask[nans] = True
            if ninf:
                data.mask[infs] = True

        data = data.compressed()

        numsum = {
            'n'      : size,
            'ngood'  : np.size(data),
            'nmiss'  : nmiss,
            'nnan'   : nnan,
            'ninf'   : ninf,
            'nfill'  : nfill,
            'min'    : None,
            'max'    : None,
            'mean'   : None,
            'median' : None,
            'var'    : None,
            'std'    : None,
        }

        try:
            if data.size:
                if data.dtype == np.dtype('S1'):
                    data = data.astype(int)

                numsum.update({
                    'min': data.min().item(),
                    'max': data.max().item(),
                    'mean': data.mean(dtype=np.float64).item(),
                    'median': np.median(data)
                    })

                if data.size > 1:
                    numsum['var'] = data.var(dtype=np.float64)
                    numsum['std'] = np.sqrt(numsum['var']).item() if numsum['var'] is not None and numsum['var'] >= 0 else None
        except:
            pass 
        
        return numsum

class ExStateDataType:
    '''Class containig methods for working with exclusive state data.

    Reads important metadata from an exclusive state variable, and collects
    the counts of each exclusive state in a data set into a StateSum.
    '''
    flag_desc_re = re.compile('^flag_(\d+)_description$')
    def __init__(self, ncvar, attributes):
        self.type_name = 'exclusiveState'
        # get flag values
        self.flag_values = []
        self.var_name = ncvar._name
        if hasattr(ncvar, 'flag_values'):
            if hasattr(ncvar.flag_values, 'tolist'):
                self.flag_values = ncvar.flag_values.tolist()
            elif re.match('(\-?\d+ )+\-?\d+', ncvar.flag_values):
                self.flag_values = list(map(int, ncvar.flag_values.split()))
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

    @staticmethod
    def matches(ncvar, attributes):
        return ( hasattr(ncvar, 'flag_values') or \
                 any(ExStateDataType.flag_desc_re.match(a) for a in ncvar.ncattrs()) ) and \
            issubclass(ncvar.dtype.type, np.integer)

    def summarize(self, data=None):
        if data is None:
            return {v: 0 for v in self.flag_values}
        
        d = data.astype(int)
        if hasattr(d, 'mask'):
            d = d.compressed()

        if not d.size:
            return {str(v): 0 for v in self.flag_values}

        u, c = np.unique(d, return_counts=True)

        s = { }
        for v in self.flag_values:
            i = np.where(u == v)
            s[str(v)] = c[i[0][0]] if i[0].size > 0 else 0

        return s

class InStateDataType:
    '''Class containig methods for working with inclusive state data.
    
    Reads important metadata from an inclusive state variable, and collects
    the counts of each inclusive state in a data set into a StateSum.
    '''
    bit_desc_re = re.compile('^bit_(\d+)_description$')
    def __init__(self, ncvar, attributes):
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
    def matches(ncvar, attributes):
        return ( hasattr(ncvar, 'flag_masks') or \
                 any(InStateDataType.bit_desc_re.match(a) for a in ncvar.ncattrs()) ) and \
            issubclass(ncvar.dtype.type, np.integer)

    def summarize(self, data=None):
        if data is None:
            return {str(m): 0 for m in self.flag_masks}

        return {str(m): np.sum(data&m).item() for m in self.flag_masks}

class QCDataType(InStateDataType):
    '''Class containing methods for working with Quality control data

    Subclass of InStateData, only difference is that variable names must 
    start with a qc_ to be identified as QC and not just inclusive state.
    '''
    qc_bit_desc_re = re.compile('qc_bit_(\d+)_description')
    def __init__(self, ncvar, attributes):
        InStateDataType.__init__(self, ncvar, attributes)

        self.type_name = 'qualityControl'
        if not self.flag_masks or not self.flag_descriptions:
            n_flags = 0
            for attr in attributes:
                match = QCDataType.qc_bit_desc_re.match(attr['name'])
                if not match: continue
                n_flags = max(n_flags, int(match.group(1)))

            self.flag_masks = [2**x for x in range(n_flags)]

            # get flag descriptions
            self.flag_descriptions = ['']*len(self.flag_masks)
            for n in range(len(self.flag_masks)):
                bit_desc = 'qc_bit_'+str(n+1)+'_description'
                if bit_desc in attributes:
                    self.flag_descriptions[n] = attributes[bit_desc]

    @staticmethod
    def matches(ncvar, attributes):
        return ncvar._name.startswith('qc_') and \
            (InStateDataType.matches(ncvar, attributes) or \
             any(QCDataType.qc_bit_desc_re.match(a['name']) for a in attributes
             ))

def data_type_of(ncvar, attributes):
    '''Determines the data type of ncvar
    '''
    data_types = (
        # TimeDataType, 
        DimlessDataType,
        QCDataType,
        ExStateDataType,
        InStateDataType,
        NumDataType
        )

    for data_type in data_types:
        if data_type.matches(ncvar, attributes):
            return data_type(ncvar, attributes)

### Data classes ------------------------------------------------------------------------------------------------------

def sum_timed_data(ncvar, summary_times, attributes, data_type=None):
    '''Summarize the data in a variable with a time dimension

    Parameters:
    ncvar: netCDF4 variable to get data from
    summary_times: 1d array with length of time dimension
                   where samples to be fed into a single summary
                   share a value in the array.
    attributes: global atts dictionary from get_attributes function
    '''
    data_type = data_type_of(ncvar, attributes) if data_type is None else data_type
    var_data = None
    try:
        var_data = ncvar[:]
    except Exception as e:
        logging.error('Unreadable variable data: %s' % (ncvar._name))
        logging.error(traceback.format_exc())
        return { }

    time_i = ncvar.dimensions.index('time')
    if time_i > 0:
        var_data = var_data.swapaxes(0, time_i)

    summaries = [ ]

    for t in map(int, np.unique(summary_times)):
        # select only the chunk at the desired time
        sample_data = var_data[summary_times == t]
        
        # flatten the array
        sample_data = sample_data.ravel()

        ## summarize the data and update the summary
        summaries.append(data_type.summarize(sample_data))

    keys = summaries[0].keys()
    return (lambda sums=summaries, keys=keys:{k: [s[k] for s in sums] for k in keys})()

def sum_untimed_data(ncvar, attributes, data_type=None):
    '''Summarize the data in a variable without a time dimension

    Parameters:
    ncvar: netCDF4 variable to get data from
    attributes: global atts dictionary from get_attributes function
    '''
    data_type = data_type_of(ncvar, attributes) if data_type is None else data_type
    var_data = None
    try:
        var_data = ncvar[:]
    except Exception as e:
        logging.error('Unreadable variable data: %s' % (ncvar._name))
        logging.error(traceback.format_exc())
        return { }

    var_data = var_data.ravel()

    return data_type.summarize(var_data)

### Higher level summarizers ------------------------------------------------------------------------------------------

def sum_attributes(group):
    try:
        return [{'name': str(k), 'val': list(v) if isinstance(v, np.ndarray) else v} for k, v in group.__dict__.items()]
    except AttributeError as e:
        logging.error('Variable: %s' % group._name)
        logging.error(traceback.format_exc())
    return [ ]

def sum_variable(ncvar, attributes, summary_times=None, metadata_only=False):
    '''Get summary information for a netCDF4 variable

    Parameters:
    ncvar: netCDF4 variable to get data from
    attributes: global atts dictionary from get_attributes function
    summary_times: 1d array with length of time dimension
                   where samples to be fed into a single summary
                   share a value in the array.
    metadata_only: if True, skip the data summary and only return header data.
    '''
    doc = {
        'name': ncvar._name,
        'attributes': sum_attributes(ncvar),
        'dimensions': ncvar.dimensions,
        'dtype': str(ncvar.dtype),
    }

    vtype = data_type_of(ncvar, attributes)

    if metadata_only:
        return doc

    try:
        data = None
        if 'time' in ncvar.dimensions:
            data = sum_timed_data(ncvar, summary_times, attributes, vtype)
        else:
            data = sum_untimed_data(ncvar, attributes, vtype)
    except:
        logging.error("Failed on variable: %s" % ncvar._name)
        raise

    doc['data'] = data

    return doc

class SumFile:

    def __init__(self, path, interval=60*60, mdonly=False):
        self.path = path
        self.interval = interval
        self.mdonly = mdonly

    def read(self):
        ncfile = Dataset(self.path)
        ncvars = ncfile.variables

        base_time = None
        if 'base_time' in ncvars:
            base_time = ncvars['base_time'][:]
        if not base_time:
            match = re.search('\.(\d{8}\.\d{6})\.', self.path)
            if match:
                datestr = match.group(1)
                d = datetime.strptime(datestr, "%Y%m%d.%H%M%S")
                base_time = time.mktime(d.timetuple())

        if base_time is None:
            raise RuntimeError('Cannot determine base_time: %s' % path)

        sample_times = None
        if 'time' in ncfile.dimensions and len(ncfile.dimensions['time']) > 0:
            if 'time' in ncvars:
                sample_times = ncvars['time'][:] + 86400*base_time//86400
            elif 'time_offset' in ncvars:
                sample_times = ncvars['time_offset'][:] + base_time

        summary_times = None
        if sample_times is None:
            logging.error('No sample time variable found for {}\n'.format(self.path))
            self.mdonly = True
        else:
            # If there are masked values in the time array then we have issues and can't go on since time
            # is a critical component here.
            if isinstance(sample_times, np.ma.MaskedArray) and not sample_times.mask.all():
                raise Exception("Invalid time data: %s" % self.path)
            self.beg = sample_times[0]
            self.end = sample_times[-1]
            summary_times = (sample_times // self.interval).astype(int)*self.interval
            self.summary_times = np.unique(summary_times)
        try:
            self.dimensions = \
                [{'name': str(k), 'length': len(v), 'unlimited': v.isunlimited()} for k, v in ncfile.dimensions.items()]
            self.attributes = sum_attributes(ncfile)
            self.variables = \
                [sum_variable(v, self.attributes, summary_times, self.mdonly) for v in ncvars.values()]
        except:
            logging.error('Failed on file: %s' % self.path)
            raise
     
        companion_prefixes = {
            'fgp'  : 'fraction of good points',
            'be'   : 'best estimate',
            'qc'   : 'quality control',
            'aqc'  : 'ancillary quality control'
        }
        for var in self.variables:
            companions = [v['name'] for v in self.variables \
                if any(p+'_'+var['name'] == v['name'] for p in companion_prefixes)]
            if companions:
                var['companions'] = companions

    def jsonify(self):
        return {
            'path'       : self.path,
            'span'       : (self.beg, self.end),
            'time'       : self.summary_times,
            'attributes' : self.attributes,
            'dimensions' : self.dimensions,
            'variables'  : self.variables,
        }



### Test ----------------------------------------------------------------------------------------------------------

datastream_re_str = '([a-z]{3})([a-z0-9]*)([A-Z]\d+)\.([a-z]\d)'
datastream_re = re.compile(datastream_re_str)
ncfile_re = re.compile(datastream_re_str+'\.(\d{4})(\d\d)(\d\d)\.(\d\d)(\d\d)(\d\d).(nc|cdf)$')

def test():
    '''Tests the summary and encoding functions for one file from every datastream in /data/datastream
    '''
    data_datastream = '/data/datastream'
    for site in os.listdir(data_datastream):
        site_path = os.path.join(data_datastream, site)
        for datastream in os.listdir(site_path):
            if not datastream_re.match(datastream): continue

            datastream_path = os.path.join(site_path, datastream)
            ncfiles = list(filter(ncfile_re.match, os.listdir(datastream_path)))

            if not ncfiles: continue

            ncfile_name = os.path.join(datastream_path, ncfiles[0])
            print(ncfile_name);
            ncfile_summary(ncfile_name, 60*60)

