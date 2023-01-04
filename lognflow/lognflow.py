import pathlib
import time
# import logging
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
                      'filesplit_cnt', 
                      'save_as',
                      'log_counter_limit'])

class lognflow:
    """ lognflow logs your workflow
    
    Indeed we need a simple lognflow that works on local machine disk.
    This is a minimalist approach to solve this problem. 
    """
    
    def __init__(self, 
                 logs_root : pathlib.Path = None,
                 log_dir : pathlib.Path = None,
                 print_text = True,
                 main_log_name = 'main_log'):
        """ lognflow construction
        
        The lognflow is an easy way to log your variables on a local disk..
        It will also count files up when they get too large. It also can save
        np.arrays in npz format which is better than csv or what not.
        
        Parameters
        ----------
            logs_root: pathlib.Path
                This is the root directory for all logs. We will use the time.time()
                to create a log directory for each instance of the lognflow. 
            log_dir: pathlib.Path
                This is the final directory path for the log files. 
            
            note:
            ^^^^^^^^
                One of the variables logs_root or log_dir must be provided.

            print_text ; bool
                If True, everything that is logged as text will be printed as well
        """
        self._init_time = time.time()

        if(log_dir is None):
            if(logs_root is None):
                logs_root = 'logs'
            self.log_dir = pathlib.Path(logs_root) / f'{int(self._init_time):d}/'
        else:
            self.log_dir = pathlib.Path(log_dir)
        if(not self.log_dir.is_dir()):
            self.log_dir.mkdir(parents = True, exist_ok = True)
        if(not self.log_dir.is_dir()):
            return
        
        self._print_text = print_text
        self._loggers_dict = {}
        self._vars_dict = {}
        self._single_var_call_cnt = 0

        self.log_name = main_log_name
    
    def rename(self, new_dir:str):
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
                         title = None, 
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
        if(title is not None):
            _logger.write(title + '\n')
        self._loggers_dict[log_name] = [_logger, 
                                       log_size_limit, 
                                       log_size, 
                                       log_file_id]
        
    def log_text(self, 
                 to_be_logged : str = '\n', 
                 log_name : str = None,
                 log_time_stamp = True,
                 print_text = None,
                 title = None, 
                 log_size_limit: int = int(1e+7),
                 time_in_file_name = True,
                 **_):
        time_time = time.time() - self._init_time

        if(log_name is None):
            log_name = self.log_name
        
        if(print_text is None):
            print_text = self._print_text
        
        if(print_text):
            if(log_time_stamp):
                print(f'T:{time_time:>6.6f}|')
            print(f' log name: {log_name}')
            print(to_be_logged)
        
        if not (log_name in self._loggers_dict.keys()):
            self._log_text_handler(log_name, 
                                   title = title, 
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

        if(log_size >= log_size_limit):
            self._log_text_handler(log_name)
            _, _, _, log_file_id = self._loggers_dict[log_name]
        return log_file_id
    
    def __call__(self, *args, **kwargs):
        self.log_text(*args, **kwargs)
                    
    def _prepare_param_dir(self, parameter_name):
        
        try:
            _ = parameter_name.split()
        except:
            self.log_text('The parameter name is not a string')
            self.log_text(f'Its type is {type(parameter_name)}')
            self.log_text(f'It is {parameter_name}')
        assert len(parameter_name.split()) == 1, \
            self.log_text(\
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
            self.log_text(f'Creating directory: {param_dir.absolute()}')
            param_dir.mkdir(parents = True, exist_ok = True)
        return(param_dir, param_name)                    

    def _get_log_counter_limit(self, param, log_size_limit):
        cnt_limit = int(log_size_limit/(param.size*param.itemsize))
        return cnt_limit

    def log_var(self, parameter_name : str, parameter_value, 
                save_as='npz', log_size_limit: int = int(1e+7)):
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
                filesplit_cnt, save_as, log_counter_limit = _var
            cur_index += 1
        else:
            filesplit_cnt = time.time()
            cur_index = 0

        if(cur_index >= log_counter_limit):
            self._log_var_save(parameter_name)
            filesplit_cnt = time.time()
            cur_index = 0

        if(cur_index == 0):
            data_array = np.zeros((log_counter_limit, ) + parameter_value.shape,
                                  dtype = parameter_value.dtype)
            time_array = np.zeros(log_counter_limit)
        
        try:
            time_array[cur_index] = time_time
        except:
            self.log_text(f'current index {cur_index} cannot be used in the logger')
        if(parameter_value.shape == data_array[cur_index].shape):
            data_array[cur_index] = parameter_value
        else:
            self.log_text(\
                f'Shape of variable {parameter_name} cannot change '\
                f'from {data_array[cur_index].shape} '\
                f'to {parameter_value.shape}. Coppying from the last time.')
            data_array[cur_index] = data_array[cur_index - 1]
        self._vars_dict[parameter_name] = varinlog(data_array, 
                                                  time_array, 
                                                  cur_index,
                                                  filesplit_cnt,
                                                  save_as,
                                                  log_counter_limit)

    def _log_var_save(self, parameter_name : str):
        param_dir, param_name = self._prepare_param_dir(parameter_name)
        
        _var = self._vars_dict[parameter_name]
        if(_var.save_as == 'npz'):
            fpath = param_dir / (f'{param_name}_{_var.filesplit_cnt}.npz')
            np.savez(fpath,
                time_array = _var.time_array,
                data_array = _var.data_array)
        elif(_var.save_as == 'txt'):
            fpath = param_dir / (f'{param_name}_time_{_var.filesplit_cnt}.txt')
            np.savetxt(fpath, _var.data_array)
            fpath = param_dir / (f'{param_name}_data_{_var.filesplit_cnt}.txt')
            np.savetxt(fpath, _var.data_array)
        return fpath
    
    def log_var_flush(self):
        for parameter_name in self._vars_dict.keys():
            self._log_var_save(parameter_name)
    
    def log_single(self, parameter_name : str, 
                         parameter_value,
                         save_as = 'npy'):
        time_time = time.time() - self._init_time
        
        if isinstance(parameter_value, dict):
            save_as = 'npz'
        
        save_as = save_as.strip()
        save_as = save_as.strip('.')
        
        param_dir, param_name = self._prepare_param_dir(parameter_name)
        if(save_as == 'mat'):
            if(len(param_name) == 0):
                param_name = 'data'
        
        if(len(param_name) > 0):
            fname = f'{param_name}_{time_time}'
        else:
            fname = f'{time_time}'
            
        if(save_as == 'npy'):
            fpath = param_dir / (fname + '.npy')
            np.save(fpath, parameter_value)
        elif(save_as == 'npz'):
            fpath = param_dir / (fname + '.npz')
            np.savez(fpath, **parameter_value)
        elif(save_as == 'txt'):
            fpath = param_dir / (fname + '.txt')
            np.savetxt(fpath, parameter_value)
        elif(save_as == 'mat'):
            fpath = param_dir / (fname + '.mat')
            from scipy.io import savemat
            savemat(fpath, {f'{param_name}' :parameter_value})
        elif('torch'):
            fpath = param_dir / (fname + '.torch')
            from torch import save as torch_save
            torch_save(parameter_value.state_dict(), fpath)
        return fpath
    
    def log_animation(self, parameter_name : str, stack, 
                         interval=50, blit=False, 
                         repeat_delay = None, dpi=100):
        time_time = time.time() - self._init_time
        param_dir, param_name = self._prepare_param_dir(parameter_name)
        if(len(param_name) > 0):
            fname = f'{param_name}_{time_time}'
        else:
            fname = f'{time_time}'
                    
        fpath = param_dir / (fname +'.gif')
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

    def log_plot(self, parameter_name : str, 
                       parameter_value_list,
                       x_values = None,
                       image_format='jpeg', dpi=1200):
        time_time = time.time() - self._init_time
        param_dir, param_name = self._prepare_param_dir(parameter_name)
        if(len(param_name) > 0):
            fname = f'{param_name}_{time_time}'
        else:
            fname = f'{time_time}'
            
        fpath = param_dir / (fname + '.jpg')
        
        try:
            if not isinstance(parameter_value_list, list):
                parameter_value_list = [parameter_value_list]
                
            if(x_values is not None):
                if not isinstance(x_values, list):
                    x_values = [x_values]
            
                if( not( (len(x_values) == len(parameter_value_list)) | \
                         (len(x_values) == 1) )):
                    self.log_text(f'x_values for {parameter_name} should have ' \
                        + 'length of 1 or the same as parameters list.')
                    raise ValueError
            
            for list_cnt, parameter_value in enumerate(parameter_value_list):
                if(x_values is None):
                    plt.plot(parameter_value, '-*')
                else:
                    if(len(x_values) == len(parameter_value)):
                        plt.plot(x_values[list_cnt], parameter_value)
                    else:
                        plt.plot(x_values[0], parameter_value, '-*')
                        
            plt.savefig(fpath, format=image_format, dpi=dpi)
            plt.close()
            return fpath
        except:
            self.log_text(f'Cannot plot variable {parameter_name}.')
            return None
    
    def log_hist(self, parameter_name : str, 
                       parameter_value_list,
                       n_bins = 10,
                       image_format='jpeg', dpi=1200):
        time_time = time.time() - self._init_time
        param_dir, param_name = self._prepare_param_dir(parameter_name)
        if(len(param_name) > 0):
            fname = f'{param_name}_{time_time}'
        else:
            fname = f'{time_time}'
            
        fpath = param_dir / (fname + '.jpg')
        
        try:
            if not isinstance(parameter_value_list, list):
                parameter_value_list = [parameter_value_list]
                
            for list_cnt, parameter_value in enumerate(parameter_value_list):
                bins, edges = np.histogram(parameter_value, n_bins)
                plt.bar(edges[:-1], bins, width =np.diff(edges).mean(), alpha=0.5)
                plt.plot(edges[:-1], bins)
                        
            plt.savefig(fpath, format=image_format, dpi=dpi)
            plt.close()
            return fpath
        except:
            self.log_text(f'Cannot plot variable {parameter_name}.')
            return None
    
    def log_scatter3(self, parameter_name : str, 
                       parameter_value,
                       image_format='jpeg', dpi=1200):
        time_time = time.time() - self._init_time
        param_dir, param_name = self._prepare_param_dir(parameter_name)
        if(len(param_name) > 0):
            fname = f'{param_name}_{time_time}'
        else:
            fname = f'{time_time}'
            
        fpath = param_dir / (fname + '.jpg')
        
        try:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(parameter_value[0], parameter_value[1], parameter_value[2])
            plt.savefig(fpath, format=image_format, dpi=dpi)
            plt.close()
            return fpath
        except:
            self.log_text(f'Cannot plot variable {parameter_name}.')
            return None
    
    def log_plt(self, 
                parameter_name : str, 
                image_format='jpeg', dpi=1200):
        time_time = time.time() - self._init_time
        param_dir, param_name = self._prepare_param_dir(parameter_name)
        if(len(param_name) > 0):
            fname = f'{param_name}_{time_time}'
        else:
            fname = f'{time_time}'
            
        fpath = param_dir / (fname + '.jpg')
        
        try:
            plt.savefig(fpath, format=image_format, dpi=dpi)
            plt.close()
            return fpath
        except:
            self.log_text(f'Cannot save the plt instance {parameter_name}.')
            return None
    
    def log_hexbin(self, parameter_name : str, parameter_value,
                   gridsize = 20, image_format='jpeg', dpi=1200):
        
        time_time = time.time() - self._init_time
        param_dir, param_name = self._prepare_param_dir(parameter_name)
        if(len(param_name) > 0):
            fname = f'{param_name}_{time_time}'
        else:
            fname = f'{time_time}'
        fpath = param_dir / (f'{fname}.{image_format}')
        
        plt.figure()
        plt.hexbin(parameter_value[0], 
                   parameter_value[1], 
                   gridsize = gridsize)
        plt.savefig(fpath, format=image_format, dpi=dpi)
        plt.close()    
        return fpath
    
    def log_imshow(self, parameter_name : str, parameter_value,
                   image_format='jpeg', dpi=1200):
        time_time = time.time() - self._init_time
        param_dir, param_name = self._prepare_param_dir(parameter_name)
        if(len(param_name) > 0):
            fname = f'{param_name}_{time_time}'
        else:
            fname = f'{time_time}'
            
        fpath = param_dir / (f'{fname}.{image_format}')
        
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
            parameter_value = parameter_value.reshape(new_shape[0] * new_shape[1],
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
            plt.imshow(parameter_value)
            plt.colorbar()
            plt.savefig(fpath, format = image_format, dpi=dpi)
            plt.close()
            return fpath
        else:
            self.log_text(f'Cannot plot variable {parameter_name} with shape' + \
                          f'{parameter_value.shape}')
            return

    def _handle_images_stack(self, stack, nan_borders):
        if(len(stack.shape) == 2):
            canv = np.expand_dims(stack, axis=0)
        elif(len(stack.shape) == 3):
            canv = stack
        elif((len(stack.shape) == 4) | (len(stack.shape) == 5)):
            if(len(stack.shape) == 4):
                n_imgs, n_R, n_C, n_ch = stack.shape
                if(n_ch == 3):
                    canv = stack
            if(len(stack.shape) == 5):
                n_imgs, n_R, n_C, is_rgb, n_ch = stack.shape
                if(is_rgb != 3):
                    return None
            square_side = int(np.ceil(np.sqrt(n_ch)))
            new_n_R = n_R * square_side
            new_n_C = n_C * square_side
            if(len(stack.shape) == 4):
                canv = np.zeros((n_imgs, new_n_R, new_n_C), dtype = stack.dtype)
            if(len(stack.shape) == 5):
                canv = np.zeros((n_imgs, new_n_R, new_n_C, 3), dtype = stack.dtype)
            used_ch_cnt = 0

            stack[:, :1] = np.nan
            stack[:, :, :1] = np.nan
            stack[:, -1:] = np.nan
            stack[:, :, -1:] = np.nan
            
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

    def prepare_stack_of_images(self, list_of_stacks, nan_borders = True):        
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
                   dpi=600):
        
        time_time = time.time() - self._init_time
        param_dir, param_name = self._prepare_param_dir(parameter_name)
        if(len(param_name) > 0):
            fname = f'{param_name}_{time_time}'
        else:
            fname = f'{time_time}'
        fpath = param_dir / (f'{fname}.{image_format}')
                
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
                vmin = data_canvas.min()
                vmax = data_canvas.max()
                if(list_of_masks is not None):
                    mask = list_of_masks[stack_cnt]
                    if(mask is not None):
                        if(data_canvas.shape == mask.shape):
                            data_canvas[mask==0] = 0
                            vmin = data_canvas[mask>0].min()
                            vmax = data_canvas[mask>0].max()
                        elif(not canvas_mask_warning):
                            self.log_text(\
                                'The mask shape is different from the canvas.' \
                                + ' No mask will be applied.')
                            canvas_mask_warning = True
                im = ax1.imshow(data_canvas, vmin = vmin, vmax = vmax)
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
        
        plt.savefig(fpath, format=image_format, dpi=dpi)
        plt.close()
        return fpath

    def log_confusion_matrix(self,
                             parameter_name : str,
                             cm,
                             target_names = None,
                             title='Confusion matrix',
                             cmap=None,
                             figsize = None,
                             image_format = 'jpeg',
                             dpi = 1200):
        """
        given a sklearn confusion matrix (cm), make a nice plot
    
        Arguments
        ---------
        cm:           confusion matrix from sklearn.metrics.confusion_matrix
    
        target_names: given classification classes such as [0, 1, 2]
                      the class names, for example: ['high', 'medium', 'low']
    
        title:        the text to display at the top of the matrix
    
        cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                      (http://matplotlib.org/examples/color/colormaps_reference.html)
                      plt.get_cmap('jet') or plt.cm.Blues
    
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
        time_time = time.time() - self._init_time
        param_dir, param_name = self._prepare_param_dir(parameter_name)
        if(len(param_name) > 0):
            fname = f'{param_name}_{time_time}'
        else:
            fname = f'{time_time}'
        fpath = param_dir / (f'{fname}.{image_format}')
        
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
        plt.savefig(fpath, format=image_format, dpi=dpi)
        plt.close()
        return fpath
    
    def __del__(self):
        self.log_var_flush()
        
        
def select_directory(start_directory = './'):
    from PyQt5.QtWidgets import QFileDialog, QApplication
    _ = QApplication([])
    log_dir = QFileDialog.getExistingDirectory(
        None, "Open a folder", start_directory, QFileDialog.ShowDirsOnly)
    return(log_dir)