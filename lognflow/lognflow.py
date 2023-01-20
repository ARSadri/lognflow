""" lognflow

lognflow makes logging and vieweing the logs easy in Python. It is so simple
you can code it yourself, so, why would you?!

First you give it a root to make the log directory in it or give it the
directory itself. Then start dumping data by giving the variable name and the
data with the type and you are set. 

Multiple processes in parallel can make as many instances as they want and
the logviewer can be accessed via HTTP. Because the logs are on a HDD.

There is an option to keep the logged variables in memory for a long time and
then dump them when they reach a ceratin size. This reduces the network load.
"""

import pathlib
import time
import numpy as np
import itertools
from os import sep as os_sep

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import animation

from collections import namedtuple

varinlog = namedtuple('varinlog',
                      ['data_array', 
                      'time_array', 
                      'cur_index', 
                      'file_start_time', 
                      'save_as',
                      'log_counter_limit'])

class lognflow:
    """ lognflow logs your workflow
    
    Indeed we need a simple log maker that works on local machine disk.
    This is a minimalist approach to solve this problem. 
    """
    
    def __init__(self, 
                 logs_root : pathlib.Path = None,
                 log_dir : pathlib.Path = None,
                 print_text = True,
                 main_log_name = 'main_log',
                 log_flush_period = 60,
                 log_flush_period_increase_rate = 4):
        """ lognflow initialization
        
        The lognflow is an easy way to log your variables into a directory.
        This directory is assumed to be local but it can be a map to a network
        location. It will also size up files when they get too large.
        It also can save np.arrays in npz format which is better than 
        other formats.
        
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
                
        :param log_flush_period_increase_rate:
            we do not begin the timeout for the flush to be log_flush_period
            because it can happen that a code finishes very fast. Such a 
            code of course will flush everything when Python closes all
            the files. We would like to flush at small intervals at first
            and increase that timput over time. This is the rate. However,
            when the log_flush_period reaches the given input, it will
            not increase anymore.
        :type log_flush_period_increase_rate: int
        
        """
        self._init_time = time.time()

        if(log_dir is None):
            if(logs_root is None):
                logs_root = 'logs'
            self.log_dir = \
                pathlib.Path(logs_root) / f'{int(self._init_time):d}/'
        else:
            self.log_dir = pathlib.Path(log_dir)
        if(not self.log_dir.is_dir()):
            self.log_dir.mkdir(parents = True, exist_ok = True)
        assert self.log_dir.is_dir(), \
            f'Could not access the log directory {self.log_dir}'
        
        self._print_text = print_text
        self._loggers_dict = {}
        self._vars_dict = {}
        self._single_var_call_cnt = 0

        self.log_name = main_log_name
        self.last_log_flush_time = 0
        self.log_flush_period_var = 0
        self.log_flush_period_increase_rate = 4
        self.log_flush_period = log_flush_period
    
    def rename(self, new_dir:str):
        """ renaming the log directory
            It is possible to rename the log directory while logging is going
            on. This is particulary useful when at the end of an experiment,
            it is necessary to put some variables in the name of the directory,
            which is very realistic in the eyes of an experimentalist.
            
            There is only one input and that is the new name of the directory.
            
            :param new_dir: The new name of the directory.
            :type new_dir: str
            
        """
        for log_name in self._loggers_dict.keys():
            _logger, _, _, _ = self._loggers_dict[log_name]
            _logger.close()
        self.log_dir = self.log_dir.rename(self.log_dir.parent / new_dir)
        for log_name in self._loggers_dict.keys():
            _logger, log_size_limit, log_size, log_file_id = \
                self._loggers_dict[log_name]
            log_fpath = self.log_dir / log_file_id
            _logger = open(log_fpath, 'a')
            self._loggers_dict[log_name] = [_logger, 
                                       log_size_limit, 
                                       log_size, 
                                       log_file_id]
    
    def _log_text_handler(self, log_name = None, 
                         log_size_limit: int = int(1e+7),
                         time_in_file_name = True):

        if(log_name in self._loggers_dict.keys()):
            _logger, log_size_limit, log_size, log_file_id = \
                self._loggers_dict[log_name]
            _logger.close()
        
        log_size = 0
        if(time_in_file_name):
            log_file_id = log_name + f'_{int(time.time()):d}.txt'
        else:
            log_file_id = log_name + '.txt'
        log_fpath = self.log_dir / log_file_id
        _logger = open(log_fpath, 'w')
        self._loggers_dict[log_name] = [_logger, 
                                       log_size_limit, 
                                       log_size, 
                                       log_file_id]
        
    def log_text_flush(self):
        """ Flush the logs
            Writing text to open(file, 'a') does not constantly happen on HDD.
            There is an OS buffer in between. This funciton should be called
            regularly. lognflow calls it once in a while when log_text is
            called multiple times. but use needs to also call it once in a
            while.
            In later versions, a timer will be used to call it automatically.
        """
        time_time = time.time() - self._init_time
        self.log_flush_period_var += self.log_flush_period_increase_rate
        if(self.log_flush_period_var >  self.log_flush_period):
            self.log_flush_period_var = self.log_flush_period
        if(time_time - self.last_log_flush_time > self.log_flush_period_var):
            for log_name in self._loggers_dict.keys():
                _logger, _, _, _ = self._loggers_dict[log_name]
                _logger.flush()
            self.last_log_flush_time = time_time

    def log_text(self, 
                 log_name : str = None,
                 to_be_logged = '\n', 
                 log_time_stamp = True,
                 print_text = None,
                 log_size_limit: int = int(1e+7),
                 time_in_file_name = True):
        """ log a string into a text file
            You can shose a name for the log and give the text to put in it.
            Also you can pass a small numpy array. You can ask it to put time
            stamp in the log and in the log file name, you can disable
            printing the text. You can set the log size limit to split it into
            another file with a new time stamp.
            
            Parameters
            ----------
                log_name : str
                    examples: mylog or myscript/mylog
                    log_name can be just a name e.g. mylog, or could be a
                    pathlike name such as myscript/mylog.
                to_be_logged : str, nd.array, list, dict
                    the string to be logged, could be a list
                    or numpy array or even a dictionary. It uses str(...).
                log_time_stamp : bool
                    Put time stamp for every entry of the log
                print_text : bool
                    if False, what is logged will not be printed.
                log_size_limit : int
                    log size limit in bytes.
                time_in_file_name : bool
                    put time stamp in file names.
            
        """
        time_time = time.time() - self._init_time

        if(log_name is None):
            log_name = self.log_name
        
        if((print_text is None) | (print_text is True)):
            print_text = self._print_text
        
        if(print_text):
            if(log_time_stamp):
                print(f'T:{time_time:>6.6f}| ', end='')
            print(to_be_logged)
                
        if not (log_name in self._loggers_dict.keys()):
            self._log_text_handler(log_name, 
                                   log_size_limit = log_size_limit,
                                   time_in_file_name = time_in_file_name)

        _logger, log_size_limit, log_size, log_file_id = \
            self._loggers_dict[log_name]
        len_to_be_logged = 0
        if(log_time_stamp):
            _time_str = f'T:{time_time:>6.6f}| '
            _logger.write(_time_str)
            len_to_be_logged += len(_time_str)
        if isinstance(to_be_logged, np.ndarray):
            try:
                _logger.write('numpy.ndarray')
                len_to_be_logged += 10
                if(to_be_logged.size()>100):
                    _logger.write(', The first and last 50 elements:\n')
                    len_to_be_logged += 30
                    to_be_logged = to_be_logged.ravel()
                    _logstr = np.array2string(to_be_logged[:50])
                    len_to_be_logged += len(_logstr)
                    _logger.write(_logstr)
                    _logger.write(' ... ')
                    _logstr = np.array2string(to_be_logged[-50:])
                    _logger.write(_logstr)
                    len_to_be_logged += len(_logstr)
                else:
                    _logstr = ':\n' + np.array2string(to_be_logged)
                    len_to_be_logged += len(_logstr)
                    _logger.write(_logstr)
            except:
                _logger.write(' not possible to log ' + log_name + '\n')
                len_to_be_logged += 20
        else:
            if(isinstance(to_be_logged, list)):
                for _ in to_be_logged:
                    _tolog = str(_)
                    len_to_be_logged += len(_tolog)
                    _logger.write(_tolog)
            else:
                _tolog = str(to_be_logged)
                _logger.write(_tolog)
                len_to_be_logged += len(_tolog)
            _logger.write('\n')
        log_size += len_to_be_logged
        self._loggers_dict[log_name] = [_logger, 
                                       log_size_limit, 
                                       log_size, 
                                       log_file_id]

        self.log_text_flush()        

        if(log_size >= log_size_limit):
            self._log_text_handler(log_name)
            _, _, _, log_file_id = self._loggers_dict[log_name]
        return log_file_id
    
    def __call__(self, *args, **kwargs):
        if(len(args) == 1):
            self.log_text(self.log_name,
                          to_be_logged = args[0], 
                          **kwargs)
        else:
            self.log_single(*args, **kwargs)
                    
    def _prepare_param_dir(self, parameter_name):
        try:
            _ = parameter_name.split()
        except:
            self.log_text(
                self.log_name,
                'The parameter name is not a string. ' \
                + f'Its type is {type(parameter_name)}. It is {parameter_name}')
        assert len(parameter_name.split()) == 1, \
            self.log_text(self.log_name,\
                  f'The variable name {parameter_name} you chose is splitable' \
                + f' I can split it into {parameter_name.split()}'             \
                + ' Make sure you dont use space, tab, or ....'                \
                + ' If you are using single backslash, e.g. for windows'       \
                + ' folders, replace it with \\ or make it a literal string'   \
                + ' by putting an r before the variable name.')
        
        is_dir = (parameter_name[-1] == '/') | (parameter_name[-1] == '\\') \
                 | (parameter_name[-1] == r'/') | (parameter_name[-1] == r'\\')
        param_dir = self.log_dir /  parameter_name
        
        if(is_dir):
            param_name = ''
        else:
            param_name = param_dir.name
            param_dir = param_dir.parent
        if(not param_dir.is_dir()):
            self.log_text(self.log_name,
                          f'Creating directory: {param_dir.absolute()}')
            param_dir.mkdir(parents = True, exist_ok = True)
        return(param_dir, param_name)                    

    def _get_log_counter_limit(self, param, log_size_limit):
        cnt_limit = int(log_size_limit/(param.size*param.itemsize))
        return cnt_limit

    def log_var(self, parameter_name : str, parameter_value, 
                save_as='npz', log_size_limit: int = int(1e+7)):
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
            
            Parameters
            ----------
                parameter_name : str
                    examples: myvar or myscript/myvar
                    parameter_name can be just a name e.g. myvar, or could be a
                    path like name such as myscript/myvar.
                parameter_value : np.array
                    An np array whose size doesn't change
                save_as : str
                    can be 'npz' or 'txt' which will save it as text.
                log_size_limit: int
                    log_size_limit in bytes, default : 1e+7.
                    
        """
        
        time_time = time.time() - self._init_time
        
        try:
            _ = parameter_value.shape
        except:
            parameter_value = np.array([parameter_value])
        
        log_counter_limit = self._get_log_counter_limit(\
            parameter_value, log_size_limit)

        if(parameter_name in self._vars_dict.keys()):
            _var = self._vars_dict[parameter_name]
            data_array, time_array, cur_index, \
                file_start_time, save_as, log_counter_limit = _var
            cur_index += 1
        else:
            file_start_time = time.time()
            cur_index = 0

        if(cur_index >= log_counter_limit):
            self._log_var_save(parameter_name)
            file_start_time = time.time()
            cur_index = 0

        if(cur_index == 0):
            data_array = np.zeros((log_counter_limit, ) + parameter_value.shape,
                                  dtype = parameter_value.dtype)
            time_array = np.zeros(log_counter_limit)
        
        try:
            time_array[cur_index] = time_time
        except:
            self.log_text(
                self.log_name,
                f'current index {cur_index} cannot be used in the logger')
        if(parameter_value.shape == data_array[cur_index].shape):
            data_array[cur_index] = parameter_value
        else:
            self.log_text(
                self.log_name,
                f'Shape of variable {parameter_name} cannot change '\
                f'from {data_array[cur_index].shape} '\
                f'to {parameter_value.shape}. Coppying from the last time.')
            data_array[cur_index] = data_array[cur_index - 1]
        self._vars_dict[parameter_name] = varinlog(data_array, 
                                                  time_array, 
                                                  cur_index,
                                                  file_start_time,
                                                  save_as,
                                                  log_counter_limit)

    def _log_var_save(self, parameter_name : str):
        param_dir, param_name = self._prepare_param_dir(parameter_name)
        
        _var = self._vars_dict[parameter_name]
        if(_var.save_as == 'npz'):
            fpath = param_dir / f'{param_name}_{_var.file_start_time}.npz'
            np.savez(fpath,
                time_array = _var.time_array,
                data_array = _var.data_array)
        elif(_var.save_as == 'txt'):
            fpath = param_dir / f'{param_name}_time_{_var.file_start_time}.txt'
            np.savetxt(fpath, _var.time_array)
            fpath = param_dir / f'{param_name}_data_{_var.file_start_time}.txt'
            np.savetxt(fpath, _var.data_array)
        return fpath
    
    def log_var_flush(self):
        """ Flush the buffered numpy arrays
            If you have been using log_ver, this will flush all the buffered
            arrays. It is called using log_size_limit for a variable and als
            when the code that made the logger ends.
        """
        for parameter_name in self._vars_dict.keys():
            self._log_var_save(parameter_name)
    
    def _get_fpath(self, 
                   param_dir, param_name, 
                   save_as, time_in_file_name):
        time_time = time.time() - self._init_time
        if(save_as == 'mat'):
            if(len(param_name) == 0):
                param_name = param_dir.name
        
        if(len(param_name) > 0):
            fname = f'{param_name}'
            if(time_in_file_name):
                fname += f'_{time_time}'
        else:
            fname = f'{time_time}'
            
        return(param_dir / f'{fname}.{save_as}')
    
    def log_single(self, parameter_name : str, 
                         parameter_value,
                         save_as = 'npy',
                         mat_field = None,
                         time_in_file_name = True):
        """log a single variable
            The most frequently used function would probably be this one.
            
            if you call the logger object as a function and give it a parameter
            name and something to be logged, the __call__ referes to this
            function.
            
            Parameters
            ----------
                parameter_name : str
                    examples: myvar or myscript/myvar
                    parameter_name can be just a name e.g. myvar, or could be a
                    path like name such as myscript/myvar.
                parameter_value : np.array
                    An np array whose size doesn't change
                save_as : str
                    can be 'npz' or 'txt' which will save it as text.
                mat_field : str
                    when saving as 'mat' file, the field can be set.
                    otherwise it will be the parameter_name
                time_in_file_name: bool
                    Wheather if the time stamp is in the file name or not.
                    
        """
        if ((save_as is None) & isinstance(parameter_value, dict)):
            save_as = 'npz'
        
        save_as = save_as.strip()
        save_as = save_as.strip('.')
        
        param_dir, param_name = self._prepare_param_dir(parameter_name)
        fpath = self._get_fpath(param_dir, param_name, 
                                save_as, time_in_file_name)
            
        if(save_as == 'npy'):
            np.save(fpath, parameter_value)
        elif(save_as == 'npz'):
            np.savez(fpath, **parameter_value)
        elif(save_as == 'txt'):
            with open(fpath,'a') as fdata: 
                fdata.write(str(parameter_value))
        elif(save_as == 'mat'):
            from scipy.io import savemat
            if(mat_field is None):
                mat_field = param_name
            savemat(fpath, {f'{mat_field}' :parameter_value})
        elif(save_as == 'torch'):
            from torch import save as torch_save
            torch_save(parameter_value.state_dict(), fpath)
        return fpath
    
    def log_plt(self, 
                parameter_name : str, 
                image_format='jpeg', dpi=1200,
                time_in_file_name = True):
        """log a single plt
            log a plt that you have on the screen.
            
            Parameters
            ----------
                parameter_name : str
                    examples: myvar or myscript/myvar
                    parameter_name can be just a name e.g. myvar, or could be a
                    path like name such as myscript/myvar.
                time_in_file_name: bool
                    Wheather if the time stamp is in the file name or not.
                    
        """
        param_dir, param_name = self._prepare_param_dir(parameter_name)
        fpath = self._get_fpath(param_dir, param_name, 
                                image_format, time_in_file_name)
        
        try:
            plt.savefig(fpath, format=image_format, dpi=dpi)
            plt.close()
            return fpath
        except:
            plt.close()
            self.log_text(self.log_name,
                          f'Cannot save the plt instance {parameter_name}.')
            return None
    
    def log_animation(self, parameter_name : str, stack, 
                         interval=50, blit=False, 
                         repeat_delay = None, dpi=100,
                         time_in_file_name = True):
        
        """Make an animation from a stack of images
            
            Parameters
            ----------
                parameter_name : str
                    examples: myvar or myscript/myvar
                    parameter_name can be just a name e.g. myvar, or could be a
                    path like name such as myscript/myvar.
                stack : np.array of shape n_f x n_r x n_c or n_f x n_r x n_c x 3
                    stack[cnt] needs to be plotable by plt.imshow()
                time_in_file_name: bool
                    Wheather if the time stamp is in the file name or not.
        """
        fig, ax = plt.subplots()
        ims = []
        for img in stack:    
            im = ax.imshow(img, animated=True)
            plt.xticks([]),plt.yticks([])
            ims.append([im])
        ani = animation.ArtistAnimation(\
            fig, ims, interval = interval, blit = blit,
            repeat_delay = repeat_delay)    

        param_dir, param_name = self._prepare_param_dir(parameter_name)
        fpath = self._get_fpath(param_dir, param_name, 'gif', time_in_file_name)
        ani.save(fpath, dpi = dpi, 
                 writer = animation.PillowWriter(fps=int(1000/interval)))
        return fpath

    def log_plot(self, parameter_name : str, 
                       parameter_value_list,
                       x_values = None,
                       image_format='jpeg', dpi=1200,
                       time_in_file_name = True):
        """log a single plot
            If you have a numpy array or a list of arrays (or indexable by
            first dimension, an array of 1D arrays), use this to log a plot 
            
            Parameters
            ----------
                parameter_name : str
                    examples: myvar or myscript/myvar
                    parameter_name can be just a name e.g. myvar, or could be a
                    path like name such as myscript/myvar.
                parameter_value_list : np.array
                    An np array or a list of np arrays or indexable-by-0th-dim
                    np arrays
                x_values : np.array
                    if set, must be an np.array of same size of all y values
                    or a list for each vector in y values where every element
                    of x-values list is the same as the y-values element in 
                    their list
                time_in_file_name: bool
                    Wheather if the time stamp is in the file name or not.
                    
        """
        
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
                    plt.plot(parameter_value, '-*')
                else:
                    if(len(x_values) == len(parameter_value)):
                        plt.plot(x_values[list_cnt], parameter_value)
                    else:
                        plt.plot(x_values[0], parameter_value, '-*')
            
            fpath = self.log_plt(
                parameter_name = parameter_name, 
                image_format=image_format, dpi=dpi,
                time_in_file_name = time_in_file_name)
                        
            return fpath
        except:
            self.log_text(self.log_name,
                          f'Cannot plot variable {parameter_name}.')
            return None
    
    def log_hist(self, parameter_name : str, 
                       parameter_value_list,
                       n_bins = 10,
                       alpha = 0.5,
                       image_format='jpeg', dpi=1200,
                       time_in_file_name = True):
        """log a single histogram
            If you have a numpy array or a list of arrays (or indexable by
            first dimension, an array of 1D arrays), use this to log a hist
            if multiple inputs are given they will be plotted on top of each
            other using the alpha opacity. 
            
            Parameters
            ----------
                parameter_name : str
                    examples: myvar or myscript/myvar
                    parameter_name can be just a name e.g. myvar, or could be a
                    path like name such as myscript/myvar.
                parameter_value_list : np.array
                    An np array or a list of np arrays or indexable-by-0th-dim
                    np arrays
                n_bins : number or np.array
                    used to set the bins for making of the histogram
                alpha : float 
                    the opacity of histograms, a flot between 0 and 1. If you
                    have multiple histograms on top of each other,
                    use 1/number_of_your_variables.
                time_in_file_name: bool
                    Wheather if the time stamp is in the file name or not.
                    
        """
        try:
            if not isinstance(parameter_value_list, list):
                parameter_value_list = [parameter_value_list]
                
            for list_cnt, parameter_value in enumerate(parameter_value_list):
                bins, edges = np.histogram(parameter_value, n_bins)
                plt.bar(edges[:-1], bins, 
                        width =np.diff(edges).mean(), alpha=alpha)
                plt.plot(edges[:-1], bins)
            
            fpath = self.log_plt(
                parameter_name = parameter_name, 
                image_format=image_format, dpi=dpi,
                time_in_file_name = time_in_file_name)
            return fpath
        except:
            self.log_text(self.log_name,
                          f'Cannot plot variable {parameter_name}.')
            return None
    
    def log_scatter3(self, parameter_name : str, 
                       parameter_value,
                       image_format='jpeg', dpi=1200,
                       time_in_file_name = True):
        """log a single scatter in 3D
            Scatter plotting in 3D
            
            Parameters
            ----------
                parameter_name : str
                    examples: myvar or myscript/myvar
                    parameter_name can be just a name e.g. myvar, or could be a
                    path like name such as myscript/myvar.
                parameter_value : np.array
                    An np array of size 3 x n, to sctter n data points in 3D
                time_in_file_name: bool
                    Wheather if the time stamp is in the file name or not.
                    
        """
        try:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(parameter_value[0], 
                       parameter_value[1], 
                       parameter_value[2])
            fpath = self.log_plt(
                parameter_name = parameter_name, 
                image_format=image_format, dpi=dpi,
                time_in_file_name = time_in_file_name)
            return fpath
        except:
            self.log_text(self.log_name,
                          f'Cannot plot variable {parameter_name}.')
            return None
    
    def log_hexbin(self, parameter_name : str, parameter_value,
                   gridsize = 20, image_format='jpeg', dpi=1200,
                   time_in_file_name = True):
        """log a 2D histogram 
            The 2D histogram is made out of hexagonals
            
            Parameters
            ----------
                parameter_name : str
                    examples: myvar or myscript/myvar
                    parameter_name can be just a name e.g. myvar, or could be a
                    path like name such as myscript/myvar.
                parameter_value : np.array
                    An np array of size 2 x n, to make the 2D histogram
                gridsize : int
                    grid size is the number of bins in 2D
                time_in_file_name: bool
                    Wheather if the time stamp is in the file name or not.
                    
        """        
        plt.figure()
        plt.hexbin(parameter_value[0], 
                   parameter_value[1], 
                   gridsize = gridsize)
        fpath = self.log_plt(
                parameter_name = parameter_name, 
                image_format=image_format, dpi=dpi,
                time_in_file_name = time_in_file_name)
        return fpath
    
    def log_imshow(self, parameter_name : str, 
                   parameter_value,
                   image_format='jpeg', dpi=1200, cmap = 'jet',
                   time_in_file_name = True):
        """log an image
            The image is logged using plt.imshow
            
            Parameters
            ----------
                parameter_name : str
                    examples: myvar or myscript/myvar
                    parameter_name can be just a name e.g. myvar, or could be a
                    path like name such as myscript/myvar.
                parameter_value : np.array
                    An np array of size n x m, to be shown by imshow
                time_in_file_name: bool
                    Wheather if the time stamp is in the file name or not.
                    
        """
        parameter_value = np.squeeze(parameter_value)
        parameter_value_shape = parameter_value.shape
        n_dims = len(parameter_value_shape)
        
        FLAG_img_ready = False
        if(n_dims == 2):
            FLAG_img_ready = True
        elif(n_dims == 3):
            if(parameter_value_shape[2] == 3):
                FLAG_img_ready = True
        elif(n_dims == 4):
            parameter_value = parameter_value.swapaxes(1,2)
            new_shape = parameter_value.shape
            parameter_value = \
                parameter_value.reshape(new_shape[0] * new_shape[1],
                                        new_shape[2] * new_shape[3])
            FLAG_img_ready = True
        elif(n_dims == 5):
            if(parameter_value_shape[4] == 3):
                parameter_value = parameter_value.swapaxes(1,2)
                new_shape = parameter_value.shape
                parameter_value = parameter_value.reshape(\
                    new_shape[0] * new_shape[1],
                    new_shape[2] * new_shape[3],
                    new_shape[4])
                FLAG_img_ready = True
        
        if(FLAG_img_ready):
            plt.imshow(parameter_value, cmap = cmap)
            plt.colorbar()
            fpath = self.log_plt(
                parameter_name = parameter_name, 
                image_format=image_format, dpi=dpi,
                time_in_file_name = time_in_file_name)
            return fpath
        else:
            plt.close()
            self.log_text(
                self.log_name,
                f'Cannot plot variable {parameter_name} with shape' + \
                f'{parameter_value.shape}')
            return

    def multichannel_to_square(self, stack, nan_borders = np.nan):
        """ turn a stack of multi-channel images into stack of square images
            This is very useful when lots of images need to be tiled
            against each other.
        
            Parameters
            ----------
                stack : np.ndarray
                It must have the shape of either
                    n_f x n_r x n_c x n_ch
                    n_f x n_r x n_c x 3 x n_ch
                    
                In both cases n_ch will be turned into square tile
                Remember if you have N images to put into a square, you only
                have n_f = 1 image with n_ch = N, you do not have N images
                and the shape of the ndarray will be 1 x n_r x n_c x N
        """
        if((len(stack.shape) == 4) | (len(stack.shape) == 5)):
            if(len(stack.shape) == 4):
                n_imgs, n_R, n_C, n_ch = stack.shape
            if(len(stack.shape) == 5):
                n_imgs, n_R, n_C, is_rgb, n_ch = stack.shape
                if(is_rgb != 3):
                    return None
            square_side = int(np.ceil(np.sqrt(n_ch)))
            new_n_R = n_R * square_side
            new_n_C = n_C * square_side
            if(len(stack.shape) == 4):
                canv = np.zeros((n_imgs, new_n_R, new_n_C), 
                                dtype = stack.dtype)
            if(len(stack.shape) == 5):
                canv = np.zeros((n_imgs, new_n_R, new_n_C, 3),
                                 dtype = stack.dtype)
            used_ch_cnt = 0

            stack[:,   :1      ] = nan_borders
            stack[:,   : ,   :1] = nan_borders
            stack[:, -1:       ] = nan_borders
            stack[:,   : , -1: ] = nan_borders
            
            for rcnt in range(square_side):
                for ccnt in range(square_side):
                    ch_cnt = rcnt + square_side*ccnt
                    if (ch_cnt<n_ch):
                        canv[:, rcnt*n_R : (rcnt + 1)*n_R,
                                ccnt*n_C : (ccnt + 1)*n_C] = \
                            stack[..., used_ch_cnt]
                        used_ch_cnt += 1
        else:
            return None
        return canv

    def _handle_images_stack(self, stack, nan_borders = np.nan):
        canv = None
        if(len(stack.shape) == 2):
            canv = np.expand_dims(stack, axis=0)
        elif(len(stack.shape) == 3):
            canv = stack
        elif((len(stack.shape) == 4) | (len(stack.shape) == 5)):
            canv = self.multichannel_to_square(stack, nan_borders = nan_borders)
        return canv
    
    def prepare_stack_of_images(self, 
                                list_of_stacks, 
                                nan_borders = np.nan):
        """Prepare the stack of images
            If you wish to use the log_canvas, chances are you have a list
            of stacks of images where one element, has many channels.
            In that case, the channels can be tiled beside each other
            to make one image for showing. This is very useful for ML apps.
    
            Parameters
            ----------
                list_of_stacks
                    list_of_stacks would include arrays iteratable by their
                    first dimension.
                nan_borders : float
                    borders between tiles will be filled with this variable
                    default: np.nan
        """        
        if (not isinstance(list_of_stacks, list)):
            list_of_stacks = [list_of_stacks]
        for cnt, stack in enumerate(list_of_stacks):
            stack = self._handle_images_stack(stack, nan_borders = nan_borders)
            if(stack is None):
                return
            list_of_stacks[cnt] = stack
        return(list_of_stacks)

    def log_canvas(self, 
                   parameter_name : str,
                   list_of_stacks : list,
                   list_of_masks = None,
                   figsize_ratio = 1,
                   text_as_colorbar = False,
                   use_colorbar = False,
                   image_format='jpeg', 
                   cmap = 'jet',
                   dpi=600,
                   time_in_file_name = True):
        """log a cavas of stacks of images
            One way to show many images and how they change is to make
            stacks of images and put them in a list. Then each
            element of the list is supposed to be iteratable by the first
            dimension and all must have the same length.
            This function will start putting them in coloumns of a canvas.
            If you have an image with many channels, call 
            prepare_stack_of_images on the list to make a large single
            image by tiling the channels of that element beside each other.
            This is very useful when it comes to self-supervised ML.
            
            Parameters
            ----------
                parameter_name : str
                    examples: myvar or myscript/myvar
                    parameter_name can be just a name e.g. myvar, or could be a
                    path like name such as myscript/myvar.
                list_of_stacks : list
                    List of stack of images, each of which can be a
                    n_F x n_r x n_c. Notice that n_F should be the same for all
                    elements of the list.
                list_of_masks : list
                    the same as the list_of_stacks and will be used to make
                    accurate colorbars
                text_as_colorbar : bool
                    if True, max and mean and min of each image will be written
                    on it.
                use_colorbar : bool
                    actual colorbar for each iamge will be shown
                time_in_file_name: bool
                    Wheather if the time stamp is in the file name or not.
                    
        """
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
        
        canvas_mask_warning = False
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
                        elif(not canvas_mask_warning):
                            self.log_text(self.log_name,\
                                'The mask shape is different from the canvas.' \
                                + ' No mask will be applied.')
                            canvas_mask_warning = True
                else:
                    data_canvas_stat = data_canvas.copy()
                data_canvas_stat = data_canvas_stat[np.isnan(data_canvas_stat) == 0]
                data_canvas_stat = data_canvas_stat[np.isinf(data_canvas_stat) == 0]
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
        
        fpath = self.log_plt(
                parameter_name = parameter_name, 
                image_format=image_format, dpi=dpi,
                time_in_file_name = time_in_file_name)
        return fpath

    def log_confusion_matrix(self,
                             parameter_name : str,
                             cm,
                             target_names = None,
                             title='Confusion matrix',
                             cmap=None,
                             figsize = None,
                             image_format = 'jpeg',
                             dpi = 1200,
                             time_in_file_name = True):
        """
        given a sklearn confusion matrix (cm), make a nice plot
    
        Parameters
        ---------
        cm:           confusion matrix from sklearn.metrics.confusion_matrix
    
        target_names: given classification classes such as [0, 1, 2]
                      the class names, for example: ['high', 'medium', 'low']
    
        title:        the text to display at the top of the matrix
    
        cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                      (http://matplotlib.org/examples/color/colormaps_reference.html)
                      plt.get_cmap('jet') or plt.cm.Blues
        
        time_in_file_name: if True, the file name will be stamped with time
    
        Usage
        -----
        plot_confusion_matrix(\
            cm           = cm,                  # confusion matrix created by
                                                # sklearn.metrics.confusion_matrix
            target_names = y_labels_vals,       # list of names of the classes
            title        = best_estimator_name) # title of graph
    
        Citiation
        ---------
        http://scikit-learn.org/stable/auto_examples/model_selection/
                                                           plot_confusion_matrix.html
    
        """
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
    
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            clr = np.array([1, 1, 1, 0]) \
                  * (cm[i, j] - cm.min()) \
                      / (cm.max() - cm.min()) + np.array([0, 0, 0, 1])
            plt.text(j, i, f"{cm[i, j]:2.02f}", horizontalalignment="center",
                     color=clr)
        
        plt.ylabel('True label')
        plt.xlabel('Predicted label\naccuracy={:0.4f}; ' \
                   + 'misclass={:0.4f}'.format(accuracy, misclass))
        plt.title(title)
        plt.colorbar(im,fraction=0.046, pad=0.04)
        plt.tight_layout()
        fpath = self.log_plt(
                parameter_name = parameter_name, 
                image_format=image_format, dpi=dpi,
                time_in_file_name = time_in_file_name)
        return fpath
    
    def __del__(self):
        self.log_text_flush()
        self.log_var_flush()
        
def select_directory(start_directory = './'):
    """ Open dialog to select a directory
        It works for windows and Linux.
    
        Parameters
        ----------
            start_directory: pathlib.Path
            When dialog opens, it starts from here.
    """
    from PyQt5.QtWidgets import QFileDialog, QApplication
    _ = QApplication([])
    log_dir = QFileDialog.getExistingDirectory(
        None, "Open a folder", start_directory, QFileDialog.ShowDirsOnly)
    return(log_dir)