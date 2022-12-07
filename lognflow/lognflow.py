import pathlib
import time
import logging
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
                 print_text = True):
        """ lognflow construction
        
        The lognflow is an easy way to log your variables on a local disk..
        It will also count files up when they get too large. It also can save
        np.arrays in npz format which is better than csv or what not.
        
        Parameters
        ----------
            logs_root: pathlib.Path
                This is the root directory for all logs. We will use the time.time()
                to create a log directory for each instance of the lognflow. 
                default: logs/
            print_text ; bool
                If True, everything that is logged as text will be printed as well
        """
        self.init_time = time.time()

        if(log_dir is None):
            if(logs_root is None):
                logs_root = 'logs'
            self.log_dir = pathlib.Path(logs_root) / f'{int(self.init_time):d}/'
        else:
            self.log_dir = pathlib.Path(log_dir)
        if(not self.log_dir.is_dir()):
            self.log_dir.mkdir(parents = True, exist_ok = True)
        if(not self.log_dir.is_dir()):
            return
        
        self.print_text = print_text

        self.loggers_dict = {}
        self.vars_dict = {}
        self.variables_dir = self.log_dir
        self.snapshots_dir = self.log_dir
        self.results_dir = self.log_dir
        self.log_name = 'main_log'
        self._log_text_handler(title = 'lognflow main log file')
        self.log_text(str(self.log_dir))
        self.log_text(\
             'lognflow log file for run ' + \
             f'{int(self.init_time):d}')
        self.log_text('-=-' * 20)
        self.single_var_call_cnt = 0
        
    def _log_text_handler(self, log_name = None, 
                         title = None, 
                         n_lines_limit: int = int(1e+7)):
        if(log_name is None):
            log_name = self.log_name
        if(log_name in self.loggers_dict.keys()):
            _logger, n_lines_limit, _, _ = \
                self.loggers_dict[log_name]
            for handler in _logger.handlers[:]:
                _logger.removeHandler(handler)
                handler.close()
        
        log_text_cnt = 0
        _logger = logging.getLogger(title)
        _logger.setLevel(logging.INFO)
        log_file_id = log_name + f'{int(time.time()):d}.txt'
        log_fpath = self.log_dir / log_file_id
        fh = logging.FileHandler(log_fpath)
        fh.setLevel(logging.INFO)
        _logger.addHandler(fh)
        
        self.loggers_dict[log_name] = [_logger, 
                                       n_lines_limit, 
                                       log_text_cnt, 
                                       log_file_id]
        
    def log_text(self, text : str = '\n', 
                 log_name : str = None,
                 print_text = None,
                 title = None, 
                 n_lines_limit: int = int(1e+7),
                 **_):
        time_time = time.time() - self.init_time

        if(log_name is None):
            log_name = self.log_name
        
        if(print_text is None):
            print_text = self.print_text
        
        if(print_text):
            print(f'T:{time_time:>6.6f}| log name: {log_name}')
            print(text)
        
        if not (log_name in self.loggers_dict.keys()):
            self._log_text_handler(log_name, 
                                   title = title, 
                                   n_lines_limit = n_lines_limit)

        _logger, n_lines_limit, log_text_cnt, log_file_id = \
            self.loggers_dict[log_name]
        if isinstance(text, np.ndarray):
            try:
                _logger.info(f'T:{time_time:>6.6f}| numpy.ndarray:\n')
                if(text.size()>100):
                    _logger.info('The first and last 50 elements:')
                    text = text.ravel()
                    _logger.info(text[:50])
                    _logger.info('...')
                    _logger.info(text[-50:])
                else:
                    _logger.info(text)
            except:
                _logger.info(f'T:{time_time:>6.6f}| not possible to log ' + 
                             log_name)
        else:
            _logger.info(f'T:{time_time:>6.6f}|')
            _logger.info(text)
        log_text_cnt += len(text)
        self.loggers_dict[log_name] = [_logger, 
                                       n_lines_limit, 
                                       log_text_cnt, 
                                       log_file_id]

        if(log_text_cnt >= n_lines_limit):
            self._log_text_handler(log_name)
            _logger, _, _, _ = self.loggers_dict[log_name]
        return _logger.handlers[0].baseFilename
    
    def __call__(self, to_be_logged, 
                 log_name : str = None,
                 print_text = None, 
                 title = None, 
                 n_lines_limit: int = int(1e+7), **_):
        if isinstance(to_be_logged, str):
            self.log_text(to_be_logged, log_name, 
                          print_text = print_text,
                          title = title, 
                          n_lines_limit = n_lines_limit)
        elif isinstance(to_be_logged, np.ndarray):
            if(to_be_logged.size() < 256):
                self.log_text(f'{to_be_logged}')
            else:
                if(log_name is not None):
                    log_name = f'single_variable_{self.single_var_call_cnt}'
                    self.log_text(f'array with sahpe {to_be_logged.shape}'\
                                   ' is given without name. I named it: '\
                                  f'{log_name}')
                    self.single_var_call_cnt += 1
                self.log_single(log_name, to_be_logged)
                    
    def _prepare_param_dir(self, parameter_name, is_dir = False):
        try:
            _ = parameter_name.split()
        except:
            self.log_text('I the parameter name is not a string')
            self.log_text(f'Its type is {type(parameter_name)}')
            self.log_text(f'It is {parameter_name}')
            parameter_name.split()
        assert len(parameter_name.split()) == 1, \
            self.log_text(\
                f'The variable name {parameter_name} you chose is'          \
                +f' I can split it into {parameter_name.split()}'
                +' splitable. Make sure you dont use space, tab, or ....'   \
                +' If you are using single backslash, e.g. for windows'     \
                +' folders, replace it with \\ or make it a literal string' \
                +' by putting an r before the variable name.')
        if(is_dir):
            param_name = pathlib.Path(parameter_name).name
            parameter_name = parameter_name + os_sep + '/_'
        
        parameter_name_as_path = pathlib.Path(parameter_name)
        param_dir = self.log_dir / parameter_name_as_path.parent
        if(not is_dir):
            param_name = parameter_name_as_path.name
        if(not param_dir.is_dir()):
            self.log_text(f'Creating directory: {param_dir.absolute()}')
            param_dir.mkdir(parents = True, exist_ok = True)
        return(param_dir, param_name)                    

    def _get_log_counter_limit(self, param, log_size_limit):
        cnt_limit = int(log_size_limit/(param.size*param.itemsize))
        return cnt_limit

    def log_var(self, parameter_name : str, parameter_value, 
                save_as='npz', log_size_limit: int = int(1e+7)):
        time_time = time.time() - self.init_time
        
        try:
            _ = parameter_value.shape
        except:
            parameter_value = np.array([parameter_value])
        
        log_counter_limit = self._get_log_counter_limit(\
            parameter_value, log_size_limit)

        if(parameter_name in self.vars_dict.keys()):
            _var = self.vars_dict[parameter_name]
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
        self.vars_dict[parameter_name] = varinlog(data_array, 
                                                  time_array, 
                                                  cur_index,
                                                  filesplit_cnt,
                                                  save_as,
                                                  log_counter_limit)

    def _log_var_save(self, parameter_name : str):
        param_dir, param_name = self._prepare_param_dir(parameter_name)
        
        _var = self.vars_dict[parameter_name]
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
        for parameter_name in self.vars_dict.keys():
            self._log_var_save(parameter_name)
    
    # # Just use logviewer
    # def log_var_plot(self, parameter_name : str, entry_index = None):
    #     _var = self.vars_dict[parameter_name]            
    #     if entry_index is None:
    #         self.log_plot(parameter_name, _var.data_array, 
    #                       x_value = _var.time_array)
    #     else:
    #         self.log_plot(parameter_name, _var.data_array[entry_index], 
    #                       x_value = _var.time_array)
    #
    # def log_var_imshow(self, parameter_name : str, entry_index = None):
    #     _var = self.vars_dict[parameter_name]            
    #     if entry_index is None:
    #         self.log_imshow(parameter_name, _var.data_array)
    #     else:
    #         self.log_imshow(parameter_name, _var.data_array[entry_index])

    def log_animation(self, parameter_name : str, stack, 
                         interval=50, blit=False, 
                         repeat_delay = None, dpi=100):
        time_time = time.time() - self.init_time
        param_dir, _ = self._prepare_param_dir(parameter_name, is_dir = True)
        fpath = param_dir / (f'{time_time}.gif')

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

    def log_single(self, parameter_name : str, 
                         parameter_value,
                         save_as = 'npy'):
        time_time = time.time() - self.init_time
        param_dir, param_name = self._prepare_param_dir(parameter_name, is_dir = True)
        if(save_as == 'npy'):
            fpath = param_dir / (f'{time_time}.npy')
            np.save(fpath, parameter_value)
        elif(save_as == 'txt'):
            fpath = param_dir / (f'{time_time}.txt')
            np.savetxt(fpath, parameter_value)
        elif(save_as == 'mat'):
            fpath = param_dir / (f'{time_time}.mat')
            from scipy.io import savemat
            savemat(fpath, {f'{param_name}' :parameter_value})
        elif('torch'):
            fpath = param_dir / (f'{time_time}.torch')
            from torch import save as torch_save
            torch_save(parameter_value.state_dict(), fpath)
        return fpath
    
    def log_plot(self, parameter_name : str, 
                       parameter_value_list,
                       x_value = None,
                       image_format='jpeg', dpi=1200):
        time_time = time.time() - self.init_time
        param_dir, _ = self._prepare_param_dir(parameter_name, is_dir = True)
        fpath = param_dir / (f'{time_time}.jpg')
        
        try:
            if not isinstance(parameter_value_list, list):
                parameter_value_list = [parameter_value_list]
            
            for parameter_value in parameter_value_list:
                if(x_value is None):
                    plt.plot(parameter_value)
                else:
                    plt.plot(x_value, parameter_value)

            plt.savefig(fpath, format=image_format, dpi=dpi)
            plt.close()
            return fpath
        except:
            self.log_text(f'Cannot plot variable {parameter_name}.')
            return None
    
    def log_hexbin(self, parameter_name : str, parameter_value,
                   gridsize = 20, image_format='jpeg', dpi=1200):
        time_time = time.time() - self.init_time
        param_dir, _ = self._prepare_param_dir(parameter_name, is_dir = True)
        fpath = param_dir / (f'{time_time}.jpg')
        
        plt.figure()
        plt.hexbin(parameter_value[0], 
                   parameter_value[1], 
                   gridsize = gridsize)
        plt.savefig(fpath, format='jpg', dpi=1200)
        plt.close()    
        return fpath
    
    def log_imshow(self, parameter_name : str, parameter_value,
                   image_format='jpeg', dpi=1200):
        time_time = time.time() - self.init_time
        param_dir, _ = self._prepare_param_dir(parameter_name, is_dir = True)
        fpath = param_dir / (f'{time_time}.{image_format}')
        
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
                parameter_value = parameter_value.reshape(new_shape[0] * new_shape[1],
                                                          new_shape[2] * new_shape[3],
                                                          new_shape[4])
                FLAG_img_ready = True
        
        if(FLAG_img_ready):
            plt.imshow(parameter_value)
            plt.colorbar()
            plt.savefig(fpath, format = image_format, dpi=1200)
            plt.close()
            return fpath
        else:
            self.log_text(f'Cannot plot variable {parameter_name} with shape' + \
                          f'{parameter_value.shape}')
            return

    def _handle_images_stack(self, stack):
        # stack = np.squeeze(stack)
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
                n_imgs, n_R, n_C, n_ch, is_rgb = stack.shape
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
            for rcnt in range(square_side):
                for ccnt in range(square_side):
                    ch_cnt = rcnt + square_side*ccnt
                    if (ch_cnt<n_ch):
                        canv[:, rcnt*n_R : (rcnt + 1)*n_R,
                                ccnt*n_C : (ccnt + 1)*n_C] = \
                            stack[:, :, :, used_ch_cnt]
                        used_ch_cnt += 1
        else:
            return None
        return canv

    def prepare_stack_of_images(self, list_of_stacks):        
        for cnt, stack in enumerate(list_of_stacks):
            stack = self._handle_images_stack(stack)
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
        time_time = time.time() - self.init_time
        param_dir, _ = self._prepare_param_dir(parameter_name, is_dir = True)
                
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
        
        fpath = param_dir / f'{time_time}.{image_format}'
        plt.savefig(fpath, format=image_format, dpi=dpi)
        plt.close()
        return fpath

    # def log_model_snapshot(self, model, model_type='torch'):
    #     time_time = time.time() - self.init_time
    #     if(not self.snapshots_dir.is_dir()):
    #         self.log_text(f'Creating directory: {self.snapshots_dir.absolute()}')
    #         self.snapshots_dir.mkdir(parents = True, exist_ok = True)
    #     if(model_type == 'torch'):
    #         fpath = self.snapshots_dir / (f'{time_time:06.6f}.model')
    #         import torch
    #         torch.save(model.state_dict(), fpath)
    #     if(model_type == 'numpy'):
    #         fpath = self.snapshots_dir / (f'{time_time:06.6f}.npy')
    #         np.savez(model, fpath)
    #     self.log_text(f'Torch snapshot: {fpath}')
    #     self.latest_snapshot_path = fpath
    #     return fpath

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
                      see http://matplotlib.org/examples/color/colormaps_reference.html
                      plt.get_cmap('jet') or plt.cm.Blues
    
        Usage
        -----
        plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                                  # sklearn.metrics.confusion_matrix
                              target_names = y_labels_vals,       # list of names of the classes
                              title        = best_estimator_name) # title of graph
    
        Citiation
        ---------
        http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    
        """
        time_time = time.time() - self.init_time
    
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
            clr = np.array([1, 1, 1, 0])*(cm[i, j] - cm.min())/(cm.max() - cm.min()) + \
                  np.array([0, 0, 0, 1])
            plt.text(j, i, f"{cm[i, j]:2.02f}", horizontalalignment="center",
                     color=clr)
        
        plt.ylabel('True label')
        plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
        plt.title(title)
        plt.colorbar(im,fraction=0.046, pad=0.04)
        plt.tight_layout()

        param_dir = self.variables_dir / (f'{parameter_name}/')
        if(not param_dir.is_dir()):
            self.log_text(f'Creating directory: {param_dir.absolute()}')
            param_dir.mkdir(parents = True, exist_ok = True)
        
        fpath = param_dir / f'{parameter_name}_{time_time}.{image_format}'
        plt.savefig(fpath, format=image_format, dpi=dpi)
        plt.close()
        return fpath

# def test_lognflow():
#     logger = lognflow(r'C:\Alireza\mcemtools_logs')
#     for _ in range(20):
#         logger.log_text(_)
#
# if __name__ == '__main__':
#     test_lognflow()