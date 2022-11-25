import re
import pathlib
import numpy as np
from scipy.io import loadmat
from matplotlib.pyplot import imread

class logviewer:
    def __init__(self,
                 exp_dir : pathlib.Path,
                 logger = print):
        self.exp_dir = pathlib.Path(exp_dir)
        self.logger = logger
        if(self.exp_dir.is_dir()):
            self.logger('Found: ', str(self.exp_dir))
        else:
            self.logger('No such directory: ', str(self.exp_dir))
        
    def get_variable(self, var_name, single_shot_index = -1, mat_file_field = None):
        var_dir = self.exp_dir / 'variables'
        var_flist = list(var_dir.glob(f'{var_name}*.*'))
        if(len(var_flist) > 0):
            var_path = var_flist[0]
            if(not var_path.is_file()):
                return
            self.logger('Loading ', var_path)
            if(var_path.suffix == '.npz'):
                buf = np.load(var_path)
                time_array = buf['time_array']
                n_logs = (time_array > 0).sum()
                time_array = time_array[:n_logs]
                data_array = buf['data_array']
                data_array = data_array[:n_logs]
                return(time_array, data_array)
        else:
            var_dir = var_dir / var_name
            if(var_dir.is_dir()):
                flist = list(var_dir.glob('*.*'))
                flist.sort()
                var_path = flist[single_shot_index]
                if(var_path.suffix == '.npy'):
                    return np.load(var_path)
                elif(var_path.suffix == '.mat'):
                    if(mat_file_field is not None):
                        return loadmat(var_path)[mat_file_field]
                    else:
                        self.logger('You need to provide the field name for the mat file')
    
    def get_images_as_stack(self, var_name, n_images : int = None):
        var_dir = self.exp_dir / var_name
        var_flist = list(var_dir.glob(f'*.*'))
        #img__5.302307367324829#
        if(len(var_flist) > 0):
            img_inds = np.zeros(len(var_flist))
            for fcnt, fpath in enumerate(var_flist):
                self.logger(fpath.stem)
                ints_in_stem = (re.findall('\d+', fpath.stem ))
                str1 = str(ints_in_stem[-2]) + '.' + str(ints_in_stem[-1])
                img_inds[fcnt] = float(str1)
            inds = np.argsort(img_inds)
            var_flist = [var_flist[ind] for ind in inds]
            if(n_images is None):
                n_images = len(var_flist)
            else:
                n_images = np.minimum(n_images, len(var_flist))
            var_flist = var_flist[:n_images]
            self.logger(f'There are {n_images} images to stack')
            img = imread(var_flist[0])
            img_set = np.zeros((n_images, ) + img.shape, dtype=img.dtype)
            self.logger(f'shape is: {img_set.shape}')
            for fcnt, fpath in enumerate(var_flist):
                img_set[fcnt] = imread(fpath)
            self.logger('Image stack is ready')
            return(img_set)
    
    def get_main_log_text(self):
        flist = list(self.exp_dir.glob('main_log*.txt'))
        flist.sort()
        n_files = len(flist)
        if (n_files>0):
            txt = ''
            for fcnt in range(n_files):
                txt += flist[fcnt].readlines()
            return txt