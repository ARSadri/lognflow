""" lognflow

lognflow makes logging easy in Python. It is so simple you can code it 
yourself, so, why would you?!

lognflow logs all files into a directory by taking care of directories and 
files names. This saves you a lot of coding and makes your code readable
 when you say::

    logger = lognflow(logs_root = 'root_for_time_tagged_log_directories')
    logger.log_single('variables/variable1', variable1)
    logger('I just logged a variable.')
    
    another_logger = lognflow(log_dir = 'specific_dir')
    another_logger.log_plot('final_plot', final_plot_is_a_np_1d_array)

The next syntax is an easy way of just logging a numpy array. It will make
a new directory within the log_dir, called variables and make a npy file
named variable1 and put variable1 in it. The third line of the code above
prints the given text to the __call__ routine in the main txt file made in 
the log_dir.

As you can see, first you give it a root (logs_root) to make 
a log directory in it or give it the directory itself (log_dir). 
Then start dumping data by giving the variable name and the data with 
the type and you are set. 

Multiple processes in parallel can make as many instances as they want.

There is an option to keep the logged variables in memory for a long time and
then dump them when they reach a ceratin size. This reduces the network load.

for this the txt logs can be buffered for a chosable amount of time and 
numpy variables that don't change size can be buffered up to a certain size
before storing into the directory using log_var(name, var).

"""
import time
import numpy                                as np
import matplotlib.pyplot                    as plt
import matplotlib.gridspec                  as gridspec
from   matplotlib          import animation
from   pathlib             import Path      as pathlib_Path
from   itertools           import product   as itertools_product
from   dataclasses         import dataclass
from   sys                 import platform  as sys_platform
from   os                  import system    as os_system
                           
from   .utils              import (repr_raw, replace_all, select_directory, 
                                   multichannel_to_frame, )
from   .plt_utils          import plt_colorbar, plt_hist, plt_surface
from   .logviewer          import logviewer

@dataclass
class varinlog:
    data_array          : np.ndarray      
    time_array          : np.ndarray    
    curr_index          : int
    file_start_time     : float          
    suffix              : str
    log_counter_limit   : int

@dataclass
class textinlog:
    to_be_logged        : str   
    log_fpath           : pathlib_Path         
    log_size_limit      : int 
    log_size            : int     
    last_log_flush_time : float
    log_flush_period    : int

class lognflow:
    """Initialization
        
        lognflow creates a directory/attribute called log_dir and puts all 
        stored data in there, if logs_root is given. Otherwise give it log_dir
        to use. If you type
        
        logger = lognflow()
        
        it will try to open a dialog to select a directory, except, it will
        get a temp directory from the os and continue. 
        
        The lognflow construction can allow setting global settings that can
        be overridden later by calling each of its methods as follows.
        
        .. note::
            One of the variables ``logs_root`` or ``log_dir`` must be given.
            if ``log_dir`` is given, ``logs_root`` is disregarded.
    
        :param logs_root: 
            This is the root directory for all logs.
            We will use the time.time() to create a log directory for each 
            instance of the lognflow. 
        :type logs_root: pathlib.Path
        
        :param log_dir: 
            This is the final directory path for the log files. 
        :type log_dir: pathlib.Path
    
        :param log_prefix:
            this string will be put before the time tag for log_dir, when
            only logs_root is given.
        :type log_prefix: str
        
        :param log_suffix:
            if given, time tag will not be used and this string will be 
            put at the end of the log_dir name.
        :type log_prefix: str
        
        :param print_text: 
            If True, everything that is logged as text will be printed as well
        :type print_text: bool
        
        :param main_log_name: 
            main log file name, by default: 'main_log'
        :type main_log_name: str
                
        :param log_flush_period:
            The period between flushing the log files into HDD. By not
            flushing, you can reduce network or HDD overhead.
        :type log_flush_period: int
                
        :param time_tag:
            File names can carry time_tags in time.time() format. This 
            is pretty much the most fundamental contribution of lognflow beside
            carrying the folders around. By default all file names will stop
            having time tag if you set it here to False here. Itherwise,
            all file names will have time tag unless stated at each logging by
            log_... functions.
        :type time_tag: bool
    """
    
    def __init__(self, 
                 logs_root        : pathlib_Path = None,
                 log_dir          : pathlib_Path = None,
                 log_dir_prefix   : str          = None,
                 log_dir_suffix   : str          = None,
                 time_tag         : bool         = True,
                 print_text       : bool         = True,
                 main_log_name    : str          = 'main_log',
                 log_flush_period : int          = 10):
        self._init_time = time.time()
        self.time_tag = time_tag
        self.log_dir_prefix = log_dir_prefix
        self.log_dir_suffix = log_dir_suffix
        
        if(log_dir is None):
            if(logs_root is None):
                from tempfile import gettempdir
                logs_root = gettempdir()
                try:
                    logs_root = select_directory(logs_root)
                except:
                    print('no logs_root was provided.'
                          + 'Could not open select_folder'
                          + f'So a folder from tmp is chosen: {logs_root}')
            new_log_dir_found = False
            while(not new_log_dir_found):
                log_dir_name = ''
                if(log_dir_prefix is not None):
                    log_dir_name = str(log_dir_prefix)
                if len(log_dir_name) > 0:
                    if log_dir_name[-1] != '_':
                        log_dir_name += '_'
                if(log_dir_suffix is None):
                    log_dir_name += f'{self._init_time}'
                else:
                    log_dir_name += f'{log_dir_suffix}'
                self.log_dir = \
                    pathlib_Path(logs_root) / log_dir_name
                if(not self.log_dir.is_dir()):
                    new_log_dir_found = True
                else:
                    self._init_time = time.time()
        else:
            self.log_dir = pathlib_Path(log_dir)
        if(not self.log_dir.is_dir()):
            self.log_dir.mkdir(parents = True, exist_ok = True)
        assert self.log_dir.is_dir(), \
            f'Could not access the log directory {self.log_dir}'
        
        self.logged = logviewer(self.log_dir, self)
        
        self._print_text = print_text
        self._loggers_dict = {}
        self._vars_dict = {}
        self._single_var_call_cnt = 0

        self.log_name = main_log_name
        self.log_flush_period = log_flush_period
    
        self.log_dir_str = str(self.log_dir.absolute())
        self.enabled = True
        
    def disable(self):
        self.enabled = False
        
    def enable(self):
        self.enabled = True
    
    def name_from_file(self, fpath):
        """ 
            Given an fpath inside the logger log_dir, 
            what would be its equivalent parameter_name?
        """
        fpath_str = str(fpath.absolute())
        log_dir_str = None
        if self.log_dir_str in fpath_str:
            log_dir_str = self.log_dir_str
        if (self.log_dir_str + '/') in fpath_str:
            log_dir_str = self.log_dir_str + '/'
        if log_dir_str:
            fpath_name = fpath_str.split(log_dir_str)[-1]
            fpath_split = fpath_name.split('.')
            return '.'.join(fpath_split[:-1])

    def copy(self, parameter_name, source, suffix = None,
             time_tag = False):
        """ copy into a new file
            Given a parameter_name, the second argument will be copied into
            the first. We will try syntaxes os_system('cp') and 'copy' for
            Windows.
            
            :param parameter_name: str
                examples: myvar or myscript/myvar
                parameter_name can be just a name e.g. myvar, or could be a
                path like name such as myscript/myvar.
            :param source: str
                if source.is_file() then it is copied into its new location.
                Otherwise, we use logger.logged.get_flist(source, suffix) to 
                obtain a list of files matching the source and copy them into
                their new location.
        """
        if not self.enabled: return
        
        try:
            if source.is_file():
                flist = [source]
        except:
            flist = self.logged.get_flist(source, suffix)
        assert flist, 'source could not be found to copy'

        param_dir, param_name, suffix = self._param_dir_name_suffix(
            parameter_name, suffix)
        
        for fpath in flist:
            if len(param_name) == 0:
                new_param_name = fpath.stem
            else:
                new_param_name = param_name
            if suffix is None:
                suffix = fpath.suffix
            fpath_dest = self._get_fpath(
                param_dir, new_param_name, suffix, time_tag)
            
            if sys_platform in ["linux", "linux2", "darwin"]:
                os_system(f'cp {fpath} {fpath_dest}')
            elif sys_platform == "win32":
                os_system(f'copy {fpath} {fpath_dest}')
            
    @property
    def time_stamp(self):
        """ Current time stamp
            Gives the time after the start of the lognflow
        """
        return time.time() - self._init_time
    
    def rename(self, new_name:str, append: bool = False):
        """ renaming the log directory
            It is possible to rename the log directory while logging is going
            on. This is particulary useful when at the end of an experiment,
            it is necessary to put some variables in the name of the directory,
            which is very realistic in the eyes of an experimentalist.
            
            There is only one input and that is the new name of the directory.

            :param new_name: The new name of the directory (without parent path)
            :type new_name: str
            
            :param append: keep the time tag for the folder and 
                append it to the right side of the new name. Default: False.
            :type append: bool
            
        """
        if not self.enabled: return
        
        self.flush_all()
        if(append):
            log_dir_name = ''
            if(self.log_dir_prefix is not None):
                log_dir_name = str(self.log_dir_prefix)
            if len(log_dir_name) > 0:
                if log_dir_name[-1] != '_':
                    log_dir_name += '_'
            if(self.log_dir_suffix is None):
                log_dir_name_with_suffix = log_dir_name + f'{self._init_time}'
            else:
                log_dir_name_with_suffix = \
                    log_dir_name + f'{self.log_dir_suffix}'
            if self.log_dir.name == log_dir_name_with_suffix:
                log_dir_name += new_name
                if log_dir_name[-1] != '_':
                    log_dir_name += '_'
                if(self.log_dir_suffix is None):
                    log_dir_name += f'{self._init_time}'
                else:
                    log_dir_name += f'{self.log_dir_suffix}'
            else:
                log_dir_name = self.log_dir.name + '_' + new_name
        else:
            log_dir_name = new_name        
        new_dir = self.log_dir.parent / log_dir_name
        try:
            self.log_dir = self.log_dir.rename(new_dir)
            for log_name in list(self._loggers_dict):
                curr_textinlog = self._loggers_dict[log_name]
                curr_textinlog.log_fpath = \
                    self.log_dir /curr_textinlog.log_fpath.name
        except:
            self.log_text(None, 'Could not rename the log_dir from:')
            self.log_text(None, f'{self.log_dir.name}')
            self.log_text(None, 'into:')
            self.log_text(None, f'{new_name}')
            self.log_text(None, 'Most probably a file was open.')
        return self.log_dir
    
    def _param_dir_name_suffix(self, parameter_name: str, suffix: str = None):
        
        assert isinstance(parameter_name, str), \
            f'The parameter name {parameter_name} is not a string.' \
            + f' It is of type {type(parameter_name)}.' \
            + 'Perhaps you forgot to pass the name of the variable first.'
        parameter_name = ''.join(
            [_ for _ in repr(repr_raw(parameter_name))  if _ != '\''])
        parameter_name = replace_all(parameter_name, ' ', '_')
        parameter_name = replace_all(parameter_name, '\\', '/')
        parameter_name = replace_all(parameter_name, '//', '/')
        
        # param_dir = self.log_dir /  parameter_name
        
        if(parameter_name[-1] == '/'):
            param_name = ''
            param_dir = parameter_name
        else:
            parameter_name_split = parameter_name.split('/')
            if len(parameter_name_split) == 1:
                param_name = parameter_name
                param_dir = ''
            else:
                param_name = parameter_name_split[-1]
                param_dir = '/'.join(parameter_name_split[:-1])
        
        if(suffix == 'mat'):
            if(len(param_name) == 0):
                param_dir_split = param_dir.split('/')
                if param_dir_split[-1] == '/':
                    param_name = param_dir_split[-2]
                else:
                    param_name = param_dir_split[-1]
                    
        if(suffix is None):
            param_name_split = param_name.split('.')
            if len(param_name_split) > 1:
                param_suffix = param_name_split[-1]
                #Here you can check if it is a valid extention
                param_name = '.'.join(param_name_split[:-1])
            else:
                param_suffix = None
        else:
            param_suffix = suffix
            param_name_split = param_name.split('.')
            if len(param_name_split) > 1:
                fname_suffix = param_name_split[-1]
                if fname_suffix == param_suffix:
                    param_name = '.'.join(param_name_split[:-1])
    
        return(param_dir, param_name, param_suffix)

    def _get_fpath(self, param_dir: pathlib_Path, param_name: str = None, 
                   suffix: str = None, time_tag: bool = None) -> pathlib_Path:
        
        time_tag = self.time_tag if (time_tag is None) else time_tag
        
        _param_dir = self.log_dir / param_dir
        
        if(not _param_dir.is_dir()):
            _param_dir.mkdir(parents = True, exist_ok = True)
            
        if(param_name is not None):
            if(len(param_name) > 0):
                fname = f'{param_name}'
                if(time_tag):
                    fname += f'_{self.time_stamp:>6.6f}'
            else:
                fname = f'{self.time_stamp:>6.6f}'
                
            if(suffix is None):
                fpath = _param_dir / f'{fname}'
            else:
                fpath = _param_dir / f'{fname}.{suffix}'
            return fpath
        else:
            return _param_dir
        
    def _get_dirnamesuffix(self, param_dir, param_name, suffix):
        log_dirnamesuffix = param_name
        if(len(param_dir) > 0):
            log_dirnamesuffix = param_dir + '/' + log_dirnamesuffix
        if(len(suffix) > 0):
            log_dirnamesuffix = log_dirnamesuffix + '.' + suffix
        return log_dirnamesuffix
            
    def _log_text_handler(self, log_name: str, 
                         log_size_limit: int = int(1e+7),
                         time_tag: bool = None,
                         log_flush_period = None,
                         suffix = None):
        
        if (log_flush_period is None):
            log_flush_period = self.log_flush_period
        param_dir, param_name, suffix = self._param_dir_name_suffix(
            log_name, suffix)
        if suffix is None:
            suffix = 'txt'
        
        log_dirnamesuffix = self._get_dirnamesuffix(
            param_dir, param_name, suffix)
        
        
        fpath = self._get_fpath(param_dir, param_name, suffix, time_tag)
        self._loggers_dict[log_dirnamesuffix] = textinlog(
            to_be_logged=[],      
            log_fpath=fpath,         
            log_size_limit=log_size_limit,    
            log_size=0,          
            last_log_flush_time=0,
            log_flush_period=log_flush_period)  

    def log_text_flush(self, log_name = None, flush = False, suffix = None):
        """ Flush the text logs
            Writing text to open(file, 'a') does not constantly happen on HDD.
            There is an OS buffer in between. This funciton should be called
            regularly. lognflow calls it once in a while when log_text is
            called multiple times. but use needs to also call it once in a
            while.
            In later versions, a timer will be used to call it automatically.
            :param flush:
                force the flush regardless of when the last time was.
                default: False
            :type flush: bool
        """
        if not self.enabled: return
        log_name = self.log_name if (log_name is None) else log_name
        
        param_dir, param_name, suffix = self._param_dir_name_suffix(
            log_name, suffix)
        if suffix is None:
            suffix = 'txt'
        log_dirnamesuffix = self._get_dirnamesuffix(
            param_dir, param_name, suffix)
        
        curr_textinlog = self._loggers_dict[log_dirnamesuffix]
        
        if((self.time_stamp - curr_textinlog.last_log_flush_time \
                                           > curr_textinlog.log_flush_period)
           | flush):
            
            with open(curr_textinlog.log_fpath, 'a+') as f:
                f.writelines(curr_textinlog.to_be_logged)
                f.flush()
            curr_textinlog.to_be_logged = []
            curr_textinlog.last_log_flush_time = self.time_stamp

    def log_text(self, 
                 log_name: str = None,
                 to_be_logged = '', 
                 log_time_stamp = True,
                 print_text = None,
                 log_size_limit: int = int(1e+7),
                 time_tag: bool = None,
                 log_flush_period: int = None,
                 flush = False,
                 end = '\n',
                 new_file = False,
                 suffix = None):
        """ log a string into a text file
            You can shose a name for the log and give the text to put in it.
            Also you can pass a small numpy array. You can ask it to put time
            stamp in the log and in the log file name, you can disable
            printing the text. You can set the log size limit to split it into
            another file with a new time stamp.
            
            :param log_name: str
                   examples: mylog or myscript/mylog
                   log_name can be just a name e.g. mylog, or could be a
                   pathlike name such as myscript/mylog.
            :param to_be_logged: str, nd.array, list, dict
                   the string to be logged, could be a list
                   or numpy array or even a dictionary. It uses str(...).
            :param log_time_stamp: bool
                   Put time stamp for every entry of the log
            :param print_text: bool
                   if False, what is logged will not be printed.
            :param log_size_limit: int
                   log size limit in bytes.
            :param time_tag: bool
                   put time stamp in file names.
            :param log_flush_period: int
                   How often flush the log in seconds, if time passes this
                   given period, it will flush the first time a text is logged,
                   or if the logger is finilized.
            :param flush: bool
                   force flush into the log file
            :param end: str
                   The last charachter for this call.
            :param new_file: bool
                   if a new file is needed. If time_tag is True, it will make
                   a new file with a new name that has a time tag. If False,
                   it closees the current text file and overwrites on it.
            :param suffix: str
                   suffix is the extension of the file name.
        """
        if not self.enabled: return
        time_tag = self.time_tag if (time_tag is None) else time_tag
        log_flush_period = self.log_flush_period \
            if (log_flush_period is None) else log_flush_period
        log_name = self.log_name if (log_name is None) else log_name

        param_dir, param_name, suffix = self._param_dir_name_suffix(
            log_name, suffix)
        if suffix is None:
            suffix = 'txt'
        log_dirnamesuffix = self._get_dirnamesuffix(
            param_dir, param_name, suffix)

        if ( (not (log_dirnamesuffix in self._loggers_dict)) or new_file):
            self._log_text_handler(log_dirnamesuffix, 
                                   log_size_limit = log_size_limit,
                                   time_tag = time_tag,
                                   suffix = suffix)

        if((print_text is None) | (print_text is True)):
            print_text = self._print_text
        if(print_text):
            if(log_time_stamp):
                print(f'T:{self.time_stamp:>6.6f}| ', end='')
            print(to_be_logged, end = end)

        curr_textinlog = self._loggers_dict[log_dirnamesuffix]
        _logger = []
        if(log_time_stamp):
            _time_str = f'T:{self.time_stamp:>6.6f}| '
            _logger.append(_time_str)
        if(isinstance(to_be_logged, list)):
            for _ in to_be_logged:
                _tolog = str(_)
                _logger.append(_tolog)
        else:
            _tolog = str(to_be_logged)
            _logger.append(_tolog)
        if(len(_logger[-1]) > 0):
            if(_logger[-1][-1] != end):
                _logger.append(end)
        else:
            _logger.append(end)
        log_size = 0
        for _logger_el in _logger:
            curr_textinlog.to_be_logged.append(_logger_el)
            log_size += len(_logger_el)
        curr_textinlog.log_size += log_size
        
        self.log_text_flush(log_dirnamesuffix, flush)        

        if(log_size >= curr_textinlog.log_size_limit):
            self._log_text_handler(
                log_dirnamesuffix, 
                log_size_limit = curr_textinlog.log_size_limit,
                time_tag = curr_textinlog.time_tag,
                suffix = suffix)
            curr_textinlog = self._loggers_dict[log_dirnamesuffix]
        return curr_textinlog.log_fpath
                        
    def _get_log_counter_limit(self, param, log_size_limit):
        cnt_limit = int(log_size_limit/(param.size*param.itemsize))
        return cnt_limit

    def log_var(self, parameter_name: str, parameter_value, 
                suffix = None, log_size_limit: int = int(1e+7)):
        """log a numpy array in buffer then dump
            It can be the case that we need to take snapshots of a numpy array
            over time. The size of the array would not change and this is hoing
            to happen frequently.
            This log_ver makes a buffer in RAM and keeps many instances of the
            array along with their time stamp and then when the size of the 
            array reaches a threhshold flushes it into HDD with a file that
            has an initial time stamp.
            The benefit of using this function over log_single is that it
            does not use the connection to the directoy all time and if that is
            on a network, there will be less overhead.
            
            :param parameter_name: str
                    examples: myvar or myscript/myvar
                    parameter_name can be just a name e.g. myvar, or could be a
                    path like name such as myscript/myvar.
            :param parameter_value: np.array
                    An np array whose size doesn't change
            :param suffix: str
                    can be 'npz' or 'txt' which will save it as text.
            :param log_size_limit: int
                    log_size_limit in bytes, default: 1e+7.
                    
        """
        if not self.enabled: return
        try:
            _ = parameter_value.shape
        except:
            parameter_value = np.array([parameter_value])
        
        param_dir, param_name, suffix = self._param_dir_name_suffix(
            parameter_name, suffix)
        if(suffix is None):
            suffix = 'npz'
        log_dirnamesuffix = self._get_dirnamesuffix(
            param_dir, param_name, suffix)
        
        log_counter_limit = self._get_log_counter_limit(\
            parameter_value, log_size_limit)

        if(log_dirnamesuffix in self._vars_dict):
            _var = self._vars_dict[log_dirnamesuffix]
            data_array, time_array, curr_index, \
                file_start_time, suffix, log_counter_limit = \
                (_var.data_array, _var.time_array, _var.curr_index, \
                    _var.file_start_time, _var.suffix, _var.log_counter_limit)
            curr_index += 1
        else:
            file_start_time = self.time_stamp
            curr_index = 0

        if(curr_index >= log_counter_limit):
            self.log_var_flush(log_dirnamesuffix)
            file_start_time = self.time_stamp
            curr_index = 0

        if(curr_index == 0):
            data_array = np.zeros((log_counter_limit, ) + parameter_value.shape,
                                  dtype = parameter_value.dtype)
            time_array = np.zeros(log_counter_limit)
        
        try:
            time_array[curr_index] = self.time_stamp
        except:
            self.log_text(
                self.log_name,
                f'current index {curr_index} cannot be used in the logger')
        if(parameter_value.shape == data_array[curr_index].shape):
            data_array[curr_index] = parameter_value
        else:
            self.log_text(
                self.log_name,
                f'Shape of variable {log_dirnamesuffix} cannot change shape '\
                f'from {data_array[curr_index].shape} '\
                f'to {parameter_value.shape}. Coppying from the last time.')
            data_array[curr_index] = data_array[curr_index - 1]
        self._vars_dict[log_dirnamesuffix] = varinlog(data_array, 
                                                      time_array, 
                                                      curr_index,
                                                      file_start_time,
                                                      suffix,
                                                      log_counter_limit)

    def log_var_flush(self, parameter_name: str, suffix: str = None):
        """ Flush the buffered numpy arrays
            If you have been using log_ver, this will flush all the buffered
            arrays. It is called using log_size_limit for a variable and als
            when the code that made the logger ends.
            :param parameter_name: str
                examples: myvar or myscript/myvar
                    parameter_name can be just a name e.g. myvar, or could be a
                    path like name such as myscript/myvar.
        """
        if not self.enabled: return
        param_dir, param_name, suffix = self._param_dir_name_suffix(
            parameter_name, suffix)
        if(suffix is None):
            suffix = 'npz'
        log_dirnamesuffix = self._get_dirnamesuffix(
            param_dir, param_name, suffix)
        
        _param_dir = self._get_fpath(param_dir)
        
        _var = self._vars_dict[log_dirnamesuffix]
        _var_data_array = _var.data_array[_var.time_array > 0]
        _var_time_array = _var.time_array[_var.time_array > 0]
        if((_var.suffix == 'npz') | (_var.suffix == 'npy')):
            fpath = _param_dir / f'{param_name}_{_var.file_start_time}.npz'
            np.savez(fpath,
                time_array = _var_time_array,
                data_array = _var_data_array)
        else:
            fpath = _param_dir / f'{param_name}_time_{_var.file_start_time}.txt'
            np.savetxt(fpath, _var_time_array)
            fpath = _param_dir / f'{param_name}_data_{_var.file_start_time}.txt'
            np.savetxt(fpath, _var_data_array)
        return fpath
    
    def get_var(self, parameter_name: str, suffix: str = None) -> tuple:
        """ Get the buffered numpy arrays
            If you need the buffered variable back.
            :param parameter_name: str
                examples: myvar or myscript/myvar
                    parameter_name can be just a name e.g. myvar, or could be a
                    path like name such as myscript/myvar.
            
            :return: 
                A tuple including two np.ndarray. The first on is 1d time
                and the second one is nd buffered data.
            :rtype: 
                tuple of two nd.arrays
        
        """
        if not self.enabled: return
        param_dir, param_name, suffix = self._param_dir_name_suffix(
            parameter_name, suffix)
        if(suffix is None):
            suffix = 'npz'
        log_dirnamesuffix = self._get_dirnamesuffix(
            param_dir, param_name, suffix)
        
        _var = self._vars_dict[log_dirnamesuffix]
        data_array = _var.data_array[_var.time_array>0].copy()
        time_array = _var.time_array[_var.time_array>0].copy()
        return(time_array, data_array)

    def log_single(self, parameter_name: str, 
                         parameter_value,
                         suffix = None,
                         mat_field = None,
                         time_tag: bool = None):
        """log a single variable
            The most frequently used function would probably be this one.
            
            if you call the logger object as a function and give it a parameter
            name and something to be logged, the __call__ referes to this
            function.
            
            :param parameter_name: str
                    examples: myvar or myscript/myvar
                    parameter_name can be just a name e.g. myvar, or could be a
                    path like name such as myscript/myvar.
            :param parameter_value: np.array
                    Could be anything and np.save will be used. If it is a
                    dictionary, np.savez will be used. As you may know, np.save
                    can save all pickalables.
            :param suffix: str
                    can be 'npz', 'npy', 'mat', 'torch' for pytorch models
                    or 'txt' or anything else which will save it as text.
                    This includes 'json', 'pdb', or ...
            :param mat_field: str
                    when saving as 'mat' file, the field can be set.
                    otherwise it will be the parameter_name
            :param time_tag: bool
                    Wheather if the time stamp is in the file name or not.
                    
        """
        if not self.enabled: return
        time_tag = self.time_tag if (time_tag is None) else time_tag

        param_dir, param_name, suffix = self._param_dir_name_suffix(
            parameter_name, suffix)
        if(suffix is None):
            if isinstance(parameter_value, (np.ndarray, int, float)):
                suffix = 'npy'
            elif (isinstance(parameter_value, dict)):
                suffix = 'npz'
            else:
                suffix = 'txt'
        fpath = self._get_fpath(param_dir, param_name, suffix, time_tag)
        
        try:
            if(suffix == 'npy'):
                np.save(fpath, parameter_value)
            elif(suffix == 'npz'):
                np.savez(fpath, **parameter_value)
            elif((suffix == 'tif') | (suffix == 'tiff')):
                from tifffile import imwrite
                imwrite(fpath, parameter_value)
            elif(suffix == 'mat'):
                from scipy.io import savemat
                if(mat_field is None):
                    mat_field = param_name
                savemat(fpath, {f'{mat_field}':parameter_value})
            elif(suffix == 'torch'):
                from torch import save as torch_save
                torch_save(parameter_value.state_dict(), fpath)
            else:
                with open(fpath,'a') as fdata: 
                    fdata.write(str(parameter_value))
        except:
            fpath = None
        return fpath
    
    def log_plt(self, 
                parameter_name: str, 
                image_format='jpeg', dpi=1200,
                time_tag: bool = None,
                close_plt = True):
        """log a single plt
            log a plt that you have on the screen.
            
            :param parameter_name: str
                    examples: myvar or myscript/myvar
                    parameter_name can be just a name e.g. myvar, or could be a
                    path like name such as myscript/myvar.
            :param time_tag: bool
                    Wheather if the time stamp is in the file name or not.
                    
        """
        if not self.enabled: return
        time_tag = self.time_tag if (time_tag is None) else time_tag
            
        param_dir, param_name, image_format = \
            self._param_dir_name_suffix(parameter_name, image_format)
        fpath = self._get_fpath(param_dir, param_name, image_format, time_tag)
        
        try:
            plt.savefig(fpath, format=image_format, dpi=dpi)
            if(close_plt):
                plt.close()
            return fpath
        except:
            if(close_plt):
                plt.close()
            self.log_text(
                None, f'Cannot save the plt instance {parameter_name}.')
            return None
     
    def log_multichannel_by_subplots(self, 
        parameter_name: str, 
        parameter_value: np.ndarray,
        frame_shape = None,
        image_format='jpeg', 
        dpi=1200, 
        time_tag: bool = None,
        colorbar = False,
        remove_axis_ticks = True,
        return_figure = False,
        **kwargs):
        """log multiple images as a tiled square
            The image is logged using plt.imshow
            
            :param parameter_name: str
                    examples: myvar or myscript/myvar
                    parameter_name can be just a name e.g. myvar, or could be a
                    path like name such as myscript/myvar.
            :param parameter_value: np.array
                    An np array of size n_r, n_c, n_ch, to be shown by imshow
                    as a square tile of side length of n_ch**0.5
            :param time_tag: bool
                    Wheather if the time stamp is in the file name or not.
                    
        """
        if not self.enabled: return
        time_tag = self.time_tag if (time_tag is None) else time_tag
            
        n_r, n_c, n_ch = parameter_value.shape
        
        if(frame_shape is None):
            n_ch_sq = int(np.ceil(n_ch ** 0.5))
            n_ch_r, n_ch_c = (n_ch_sq, n_ch_sq)
        else:
            n_ch_r, n_ch_c = frame_shape
        fig, ax = plt.subplots(n_ch_r,n_ch_c)
        if(remove_axis_ticks):
            plt.setp(ax, xticks=[], yticks=[])
        for rcnt in range(n_ch_r):
            for ccnt in range(n_ch_c):
                imcnt = ccnt + rcnt * n_ch_c
                if imcnt < n_ch:
                    im = parameter_value[:,:, imcnt]
                    im_ch = ax[rcnt, ccnt].imshow(im, **kwargs)
                    if(colorbar):
                        plt_colorbar(im_ch)
        if not return_figure:
            return self.log_plt(parameter_name = parameter_name,
                         image_format=image_format, dpi=dpi,
                         time_tag = time_tag)
        else:
            return fig
            
    def log_plot(self, parameter_name: str, 
                       parameter_value_list,
                       x_values = None,
                       image_format='jpeg', dpi=1200,
                       time_tag: bool = None,
                       **kwargs):
        """log a single plot
            If you have a numpy array or a list of arrays (or indexable by
            first dimension, an array of 1D arrays), use this to log a plot 
            
            :param parameter_name: str
                    examples: myvar or myscript/myvar
                    parameter_name can be just a name e.g. myvar, or could be a
                    path like name such as myscript/myvar.
            :param parameter_value_list: np.array
                    An np array or a list of np arrays or indexable-by-0th-dim
                    np arrays
            :param x_values: np.array
                    if set, must be an np.array of same size of all y values
                    or a list for each vector in y values where every element
                    of x-values list is the same as the y-values element in 
                    their list
            :param time_tag: bool
                    Wheather if the time stamp is in the file name or not.
                    
        """
        if not self.enabled: return
        time_tag = self.time_tag if (time_tag is None) else time_tag
            
        try:
            if not isinstance(parameter_value_list, list):
                parameter_value_list = [parameter_value_list]
                
            if(x_values is not None):
                if not isinstance(x_values, list):
                    x_values = [x_values]
            
                if( not( (len(x_values) == len(parameter_value_list)) | \
                         (len(x_values) == 1) )):
                    self.log_text(
                        self.log_name,
                        f'x_values for {parameter_name} should have'\
                        + ' length of 1 or the same as parameters list.')
                    raise ValueError
            
            for list_cnt, parameter_value in enumerate(parameter_value_list):
                if(x_values is None):
                    plt.plot(parameter_value, **kwargs)
                else:
                    if(len(x_values) == len(parameter_value)):
                        plt.plot(x_values[list_cnt], parameter_value, **kwargs)
                    else:
                        plt.plot(x_values[0], parameter_value, **kwargs)
            
            fpath = self.log_plt(
                parameter_name = parameter_name, 
                image_format=image_format, dpi=dpi,
                time_tag = time_tag)
                        
            return fpath
        except:
            self.log_text(None, f'Cannot plot variable {parameter_name}.')
            return None
    
    def log_hist(self, parameter_name: str, 
                       parameter_value_list,
                       labels_list = None,
                       n_bins = 10,
                       alpha = 0.5,
                       normalize = False,
                       image_format='jpeg', dpi=1200,
                       time_tag: bool = None, 
                       return_figure = False,
                       **kwargs):
        """log a single histogram
            If you have a numpy array or a list of arrays (or indexable by
            first dimension, an array of 1D arrays), use this to log a hist
            if multiple inputs are given they will be plotted on top of each
            other using the alpha opacity. 
            
            :param parameter_name: str
                    examples: myvar or myscript/myvar
                    parameter_name can be just a name e.g. myvar, or could be a
                    path like name such as myscript/myvar.
            :param parameter_value_list: np.array
                    An np array or a list of np arrays or indexable-by-0th-dim
                    np arrays
            :param n_bins: number or np.array
                    used to set the bins for making of the histogram
            :param alpha: float 
                    the opacity of histograms, a flot between 0 and 1. If you
                    have multiple histograms on top of each other,
                    use 1/number_of_your_variables.
            :param time_tag: bool
                    Wheather if the time stamp is in the file name or not.
                    
        """
        if not self.enabled: return
        time_tag = self.time_tag if (time_tag is None) else time_tag
            
        fig, ax = plt_hist(parameter_value_list, 
                           n_bins = n_bins, alpha = alpha, 
                           normalize = normalize, 
                           labels_list = None, **kwargs)
        
        if not return_figure:
            fpath = self.log_plt(
                parameter_name = parameter_name, 
                image_format=image_format, dpi=dpi,
                time_tag = time_tag)
            return fpath
        else:
            return fig, ax
    
    def log_scatter3(self, parameter_name: str,
                       parameter_value, image_format='jpeg', dpi=1200,
                       time_tag: bool = None, return_figure = False):
        """log a single scatter in 3D
            Scatter plotting in 3D
            
            :param parameter_name: str
                    examples: myvar or myscript/myvar
                    parameter_name can be just a name e.g. myvar, or could be a
                    path like name such as myscript/myvar.
            :param parameter_value: np.array
                    An np array of size 3 x n, to sctter n data points in 3D
            :param time_tag: bool
                    Wheather if the time stamp is in the file name or not.
                    
        """
        if not self.enabled: return
        time_tag = self.time_tag if (time_tag is None) else time_tag
            
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(parameter_value[0], 
                   parameter_value[1], 
                   parameter_value[2])
        if not return_figure:
            return self.log_plt(
                parameter_name = parameter_name, 
                image_format=image_format, dpi=dpi,
                time_tag = time_tag)
        else:
            return fig, ax
    
    
    
    def log_surface(self, parameter_name: str,
                       parameter_value, image_format='jpeg', dpi=1200,
                       time_tag: bool = None, return_figure = False, **kwargs):
        """log a surface in 3D
            surface plotting in 3D exactly similar to imshow but in 3D
            
            :param parameter_name: str
                    examples: myvar or myscript/myvar
                    parameter_name can be just a name e.g. myvar, or could be a
                    path like name such as myscript/myvar.
            :param parameter_value: np.array
                    An np array of size n x m, to plot surface in 3D
            :param time_tag: bool
                    Wheather if the time stamp is in the file name or not.
            rest of the parameters (**kwargs) will be passed to plot_surface() 
        """
        if not self.enabled: return
        time_tag = self.time_tag if (time_tag is None) else time_tag
            
        fig, ax = plt_surface(parameter_value)
        
        if not return_figure:
            fpath = self.log_plt(
                parameter_name = parameter_name, 
                image_format=image_format, dpi=dpi,
                time_tag = time_tag)
            return fpath
        else:
            return fig, ax
        
    def log_hexbin(self, parameter_name: str, parameter_value,
                   gridsize = 20, image_format='jpeg', dpi=1200,
                   time_tag: bool = None):
        """log a 2D histogram 
            The 2D histogram is made out of hexagonals
            
            :param parameter_name: str
                    examples: myvar or myscript/myvar
                    parameter_name can be just a name e.g. myvar, or could be a
                    path like name such as myscript/myvar.
            :param parameter_value: np.array
                    An np array of size 2 x n, to make the 2D histogram
            :param gridsize: int
                    grid size is the number of bins in 2D
            :param time_tag: bool
                    Wheather if the time stamp is in the file name or not.
        """
        if not self.enabled: return
        time_tag = self.time_tag if (time_tag is None) else time_tag

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.hexbin(parameter_value[0], 
                   parameter_value[1], 
                   gridsize = gridsize)
        if return_figure:
            fpath = self.log_plt(
                    parameter_name = parameter_name, 
                    image_format=image_format, dpi=dpi,
                    time_tag = time_tag)
            return fpath
        else:
            return fig, ax
        
    def log_imshow(self, parameter_name: str, parameter_value, 
                   colorbar = True, remove_axis_ticks = True,
                   image_format='jpeg', dpi=1200, cmap = 'viridis',
                   time_tag: bool = None, borders = 0, **kwargs):
        """log an image
            The image is logged using plt.imshow
            
            :param parameter_name: str
                    examples: myvar or myscript/myvar
                    parameter_name can be just a name e.g. myvar, or could be a
                    path like name such as myscript/myvar.
            :param parameter_value: np.array
                    An np array of shape amongst the following:
                    * (n, m) 
                    * (n, m, 3)
                    * (n, m, ch)
                    * (1, n, m, ch)
                    * (n, m, 3, ch)
            :param time_tag: bool
                    Wheather if the time stamp is in the file name or not.
        """
        if not self.enabled: return
        time_tag = self.time_tag if (time_tag is None) else time_tag
            
        parameter_value = parameter_value.squeeze()
        parameter_value_shape = parameter_value.shape
        n_dims = len(parameter_value_shape)
        
        FLAG_img_ready = False
        use_multichannel_to_square = False
        if(n_dims == 2):
            FLAG_img_ready = True
        elif(n_dims == 3):
            if(parameter_value_shape[2] == 3):
                FLAG_img_ready = True
            else:
                FLAG_img_ready = True
                use_multichannel_to_square = True
        elif(n_dims == 4):
            if(parameter_value_shape[2] == 3):
                FLAG_img_ready = True
                use_multichannel_to_square = True
            elif(parameter_value_shape[0] == 1):
                FLAG_img_ready = True
                use_multichannel_to_square = True
        
        if(use_multichannel_to_square):
            parameter_value = self. multichannel_to_square(
                parameter_value, borders = borders)
        if(FLAG_img_ready):

            if(not np.iscomplexobj(parameter_value)):
                fig, ax = plt.subplots()
                im = ax.imshow(parameter_value, cmap = cmap, **kwargs)
                if(colorbar):
                    plt_colorbar(im)
                if(remove_axis_ticks):
                    plt.setp(ax, xticks=[], yticks=[])
            else:
                fig, ax = plt.subplots(1, 2)
                im = ax[0].imshow(np.real(parameter_value), 
                                  cmap = cmap, **kwargs)
                if(colorbar):
                    plt_colorbar(im)
                ax[0].set_title('Real')    
                im = ax[1].imshow(np.imag(parameter_value), 
                                  cmap = cmap, **kwargs)
                if(colorbar):
                    plt_colorbar(im)
                ax[1].set_title('Imag')
                if(remove_axis_ticks):
                    plt.setp(ax[0], xticks=[], yticks=[])
                    plt.setp(ax[1], xticks=[], yticks=[])
                
            fpath = self.log_plt(
                parameter_name = parameter_name, 
                image_format=image_format, dpi=dpi,
                time_tag = time_tag)
            return fpath
        else:
            self.log_text(
                self.log_name,
                f'Cannot plot variable {parameter_name} with shape' + \
                f'{parameter_value.shape}')
            return

    def _handle_images_stack(self, stack, borders = 0):
        canv = None
        if(len(stack.shape) == 2):
            canv = np.expand_dims(stack, axis=0)
        if(len(stack.shape) == 3):
            if(stack.shape[2] == 3):
                canv = np.expand_dims(stack, axis=0)
            else:
                canv = stack
        if((len(stack.shape) == 4) | (len(stack.shape) == 5)):
            canv = multichannel_to_frame(stack, borders = borders)
        return canv
    
    def prepare_stack_of_images(self, 
                                list_of_stacks, 
                                borders = 0):
        """Prepare the stack of images
            If you wish to use the log_canvas, chances are you have a list
            of stacks of images where one element, has many channels.
            In that case, the channels can be tiled beside each other
            to make one image for showing. This is very useful for ML apps.
    
            Each element of the list can appear as either:
            n_row x n_clm if only one image is in the list 
                          for all elements of stack
            n_clm x n_ros x 3 if one RGB image is given
            n_frm x n_row x n_clm if there are n_frm images 
                                  for all elements of stack
            n_frm x n_row x n_clm x n_ch if there are multiple images to be
                                         shown. 
                                         Channels will be tiled into square
            n_frm x n_row x n_clm x n_ch x 3 if channels are in RGB
    
            :param list_of_stacks
                    list_of_stacks would include arrays iteratable by their
                    first dimension.
            :param borders: float
                    borders between tiles will be filled with this variable
                    default: np.nan
        """
        if not self.enabled: return        
        list_of_stacks = list(list_of_stacks)
        for cnt, stack in enumerate(list_of_stacks):
            stack = self._handle_images_stack(stack, borders = borders)
            if(stack is None):
                return
            list_of_stacks[cnt] = stack
        return(list_of_stacks)

    def log_canvas(self, 
                   parameter_name: str,
                   list_of_stacks: list,
                   list_of_masks = None,
                   figsize_ratio = 1,
                   text_as_colorbar = False,
                   use_colorbar = False,
                   image_format='jpeg', 
                   cmap = 'viridis',
                   dpi=1200,
                   time_tag: bool = None):
        """log a cavas of stacks of images
            One way to show many images and how they change is to make
            stacks of images and put them in a list. Then each
            element of the list is supposed to be iteratable by the first
            dimension, which should be the same size for all elements in 
            the list.
            This function will start putting them in coloumns of a canvas.
            If you have an image with many channels, call 
            prepare_stack_of_images on the list to make a large single
            image by tiling the channels of that element beside each other.
            This is very useful when it comes to self-supervised ML.
            
            Each element of the list must appear as either:
            n_frm x n_row x n_clm if there are n_frm images 
                                  for all elements of stack
            n_frm x n_row x n_clm x 3 if channels are in RGB
            
            if you have multiple images as channels such as the following,
            call the prepare_stack_of_images.
            
            :param parameter_name: str
                    examples: myvar or myscript/myvar
                    parameter_name can be just a name e.g. myvar, or could be a
                    path like name such as myscript/myvar.
            :param list_of_stacks: list
                    List of stack of images, each of which can be a
                    n_F x n_r x n_c. Notice that n_F should be the same for all
                    elements of the list.
            :param list_of_masks: list
                    the same as the list_of_stacks and will be used to make
                    accurate colorbars
            :param text_as_colorbar: bool
                    if True, max and mean and min of each image will be written
                    on it.
            :param use_colorbar: bool
                    actual colorbar for each iamge will be shown
            :param time_tag: bool
                    Wheather if the time stamp is in the file name or not.
                    
        """
        if not self.enabled: return
        time_tag = self.time_tag if (time_tag is None) else time_tag
            
        try:
            _ = list_of_stacks.shape
            list_of_stacks = [list_of_stacks]
        except:
            pass
        n_stacks = len(list_of_stacks)
        if(list_of_masks is not None):
            n_masks = len(list_of_masks)
            assert (n_masks == n_stacks), \
                f'the number of masks, {n_masks} and ' \
                + f'stacks {n_stacks} should be the same'
        
        n_imgs = list_of_stacks[0].shape[0]
                
        plt.figure(figsize = (n_imgs*figsize_ratio,n_stacks*figsize_ratio))
        gs1 = gridspec.GridSpec(n_stacks, n_imgs)
        if(use_colorbar):
            gs1.update(wspace=0.25, hspace=0)
        else:
            gs1.update(wspace=0.025, hspace=0) 
        
        for img_cnt in range(n_imgs):
            for stack_cnt in range(n_stacks):
                ax1 = plt.subplot(gs1[stack_cnt, img_cnt])
                plt.axis('on')
                ax1.set_xticklabels([])
                ax1.set_yticklabels([])
                data_canvas = list_of_stacks[stack_cnt][img_cnt].copy()
                if(list_of_masks is not None):
                    mask = list_of_masks[stack_cnt]
                    if(mask is not None):
                        if(data_canvas.shape == mask.shape):
                            data_canvas[mask==0] = 0
                            data_canvas_stat = data_canvas[mask>0]
                else:
                    data_canvas_stat = data_canvas.copy()
                data_canvas_stat = data_canvas_stat[
                    np.isnan(data_canvas_stat) == 0]
                data_canvas_stat = data_canvas_stat[
                    np.isinf(data_canvas_stat) == 0]
                vmin = data_canvas_stat.min()
                vmax = data_canvas_stat.max()
                im = ax1.imshow(data_canvas, 
                                vmin = vmin, 
                                vmax = vmax,
                                cmap = cmap)
                if(text_as_colorbar):
                    ax1.text(data_canvas.shape[0]*0,
                             data_canvas.shape[1]*0.05,
                             f'{data_canvas.max():.6f}', 
                             color = 'yellow',
                             fontsize = 2)
                    ax1.text(data_canvas.shape[0]*0,
                             data_canvas.shape[1]*0.5, 
                             f'{data_canvas.mean():.6f}', 
                             color = 'yellow',
                             fontsize = 2)
                    ax1.text(data_canvas.shape[0]*0,
                             data_canvas.shape[1]*0.95, 
                             f'{data_canvas.min():.6f}', 
                             color = 'yellow',
                             fontsize = 2)
                if(use_colorbar):
                    cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
                    cbar.ax.tick_params(labelsize=1)
                ax1.set_aspect('equal')
                ax1.axis('off')
        
        fpath = self.log_plt(
                parameter_name = parameter_name, 
                image_format=image_format, dpi=dpi,
                time_tag = time_tag)
        return fpath

    def log_confusion_matrix(self,
                             parameter_name: str,
                             cm,
                             target_names = None,
                             title='Confusion matrix',
                             cmap=None,
                             figsize = None,
                             image_format = 'jpeg',
                             dpi = 1200,
                             time_tag = False):
        """log a confusion matrix
            given a sklearn confusion matrix (cm), make a nice plot
        
            :param cm:
                confusion matrix from sklearn.metrics.confusion_matrix
            
            :param target_names: 
                given classification classes such as [0, 1, 2]
                the class names, for example: ['high', 'medium', 'low']
            
            :param title:        
                the text to display at the top of the matrix
            
            :param cmap: 
                the gradient of the values displayed from matplotlib.pyplot.cm
                (http://matplotlib.org/examples/color/colormaps_reference.html)
                plt.get_cmap('viridis') or plt.cm.Blues
                
            :param time_tag: 
                if True, the file name will be stamped with time
        
            Usage::
            -----
                from lognflow import lognflow
                logger = lognflow(log_roots or log_dir)
                logger.plot_confusion_matrix(\
                    cm           = cm,        # confusion matrix created by
                                              # sklearn.metrics.confusion_matrix
                    target_names = y_labels_vals, # list of names of the classes
                    title        = best_estimator_name) # title of graph
                        
        
            Credit
            ------
                http://scikit-learn.org/stable/auto_examples/
                    model_selection/plot_confusion_matrix.html
    
        """
        if not self.enabled: return
        accuracy = np.trace(cm) / np.sum(cm).astype('float')
        misclass = 1 - accuracy
    
        if figsize is None:
            figsize = np.ceil(cm.shape[0]/3)
    
        if target_names is None:
            target_names = [chr(x + 65) for x in range(cm.shape[0])]
    
        if cmap is None:
            cmap = plt.get_cmap('Blues')
    
        plt.figure(figsize=(4*figsize, 4*figsize))
        im = plt.imshow(cm, interpolation='nearest', cmap=cmap)
    
        if target_names is not None:
            tick_marks = np.arange(len(target_names))
            plt.xticks(tick_marks, target_names, rotation=45)
            plt.yticks(tick_marks, target_names)
    
        for i, j in itertools_product(range(cm.shape[0]), range(cm.shape[1])):
            clr = np.array([1, 1, 1, 0]) \
                  * (cm[i, j] - cm.min()) \
                      / (cm.max() - cm.min()) + np.array([0, 0, 0, 1])
            plt.text(j, i, f"{cm[i, j]:2.02f}", horizontalalignment="center",
                     color=clr)
        
        plt.ylabel('True label')
        plt.xlabel('Predicted label\naccuracy={:0.4f}; ' \
                   + 'misclass={:0.4f}'.format(accuracy, misclass))
        plt.title(title)
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.tight_layout()
        fpath = self.log_plt(
                parameter_name = parameter_name, 
                image_format=image_format, dpi=dpi,
                time_tag = time_tag)
        return fpath

    def log_animation(self, parameter_name: str, stack, 
                         interval=50, blit=False, 
                         repeat_delay = None, dpi=100,
                         time_tag: bool = None):
        
        """Make an animation from a stack of images
            
            :param parameter_name: str
                    examples: myvar or myscript/myvar
                    parameter_name can be just a name e.g. myvar, or could be a
                    path like name such as myscript/myvar.
            :param stack: np.array of shape 
                    n_f x n_r x n_c or n_f x n_r x n_c x 3
                    stack[cnt] needs to be plotable by plt.imshow()
            :param time_tag: bool
                    Wheather if the time stamp is in the file name or not.
        """
        if not self.enabled: return
        time_tag = self.time_tag if (time_tag is None) else time_tag
            
        param_dir, param_name, suffix = self._param_dir_name_suffix(
            parameter_name, 'gif')
        fpath = self._get_fpath(param_dir, param_name, suffix, time_tag)

        fig, ax = plt.subplots()
        ims = []
        for img in stack:    
            im = ax.imshow(img, animated=True)
            plt.xticks([]),plt.yticks([])
            ims.append([im])
        ani = animation.ArtistAnimation(\
            fig, ims, interval = interval, blit = blit,
            repeat_delay = repeat_delay)

        ani.save(fpath, dpi = dpi, 
                 writer = animation.PillowWriter(fps=int(1000/interval)))
        return fpath

    def flush_all(self):
        if not self.enabled: return
        for log_name in list(self._loggers_dict):
            self.log_text_flush(log_name, flush = True)
        for parameter_name in list(self._vars_dict):
            self.log_var_flush(parameter_name)

    def save(self, parameter_name: str, 
                   parameter_value,
                   suffix = None,
                   mat_field = None,
                   time_tag: bool = None):
        return self.log_single(parameter_name = parameter_name, 
                               parameter_value = parameter_value,
                               suffix = suffix,
                               mat_field = mat_field,
                               time_tag = time_tag)

    def savez(self, parameter_name: str, 
                    parameter_value,
                    time_tag: bool = None):
        return self.log_single(parameter_name = parameter_name, 
                               parameter_value = parameter_value,
                               suffix = 'npz',
                               time_tag = time_tag)

    def __call__(self, *args, **kwargs):
        """calling the object
            In the case of the following code::
                logger = lognflow()
                logger('Hello lognflow')
            The text (str(...)) will be passed to the main log text file.
        """
        self.log_text(None, *args, **kwargs)
        self.flush_all()

    def __del__(self):
        try:
            self.flush_all()
        except:
            pass
        
    def __repr__(self):
        return f'{self.log_dir}'

    def __bool__(self):
        return self.log_dir.is_dir()