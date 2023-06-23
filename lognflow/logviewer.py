import pathlib
import numpy as np

class logviewer:
    """ log viewer
        Since lognflow makes lots of files and folders, maybe it is nice
        to have a logviewer that loads those information. In this module we
        provide a set of functions for a logged object that can load variables,
        texts, file lists and etc.. Use it simply by::
 
            from lognflow import logviewer
            logged = logviewer(log_dir = 'dir_contatining_files')
            var = logged.get_single('variable_name')
    """ 
    def __init__(self,
                 log_dir : pathlib.Path,
                 logger = print):
        self.log_dir = pathlib.Path(log_dir)
        assert self.log_dir.is_dir(), \
            f'lognflow.logviewer| No such directory: '+ str(self.log_dir)
        self.logger = logger
        
    def get_text(self, log_name='main_log'):
        """ get text log files
            Given the log_name, this function returns the text therein.

            Parameters
            ----------
            :param log_name:
                the log name. If not given then it is the main log.
        """
        flist = list(self.log_dir.glob(f'{log_name}*.txt'))
        flist.sort()
        n_files = len(flist)
        if (n_files>0):
            txt = []
            for fcnt in range(n_files):
                with open(flist[fcnt]) as f_txt:
                    txt.append(f_txt.readlines())
            if(n_files == 1):
                txt = txt[0]
            return txt

    def get_single(self, var_name, single_shot_index = -1, 
                     suffix = None):
        """ get a single variable
            return the value of a saved variable.

            Parameters
            ----------
            :param var_name:
                variable name
            :param single_shot_index:
                If there are many snapshots of a variable, this input can
                limit the returned to a set of indices.
            :param suffix:
                If there are different suffixes availble for a variable
                this input needs to be set. npy, npz, mat, and torch are
                supported.
                
            .. note::
                when reading a MATLAB file, the output is a dictionary.
        """
        var_name = var_name.replace('\t', '\\t').replace('\n', '\\n')\
            .replace('\r', '\\r').replace('\b', '\\b')
        
        if suffix is None:
            if len(var_name.split('.')) > 1:
                suffix = var_name.split('.')[-1]
                name_before_suffix = var_name.split('.')[:-1]
                if((len(name_before_suffix) == 1) & 
                   (name_before_suffix[0] == '')):
                    var_name = '*'
                else:
                    var_name = ('.').join(var_name.split('.')[:-1])
            else:
                suffix = '.np*'

        suffix = suffix.strip('.')        
        assert single_shot_index == int(single_shot_index), \
                    f'single_shot_index {single_shot_index} must be an integer'
        flist = []            
        if((self.log_dir / var_name).is_file()):
            flist = [self.log_dir / var_name]
        elif((self.log_dir / f'{var_name}.{suffix}').is_file()):
            flist = [self.log_dir / f'{var_name}.{suffix}']
        else:
            _var_name = (self.log_dir / var_name).name
            _var_dir = (self.log_dir / var_name).parent
            search_patt = f'{_var_name}*.{suffix}'
            while('**' in search_patt):
                search_patt = search_patt.replace('**', '*')
            flist = list(_var_dir.glob(search_patt))
            if(len(flist) == 0):
                flist = list(_var_dir.glob(f'{_var_name}*.*'))
                if(len(flist) > 0):
                    self.logger('Can not find the file with the given suffix, '\
                                +'but found some with a different suffix, '\
                                +f'one file is: {flist[single_shot_index]}')
                    
        if(len(flist) > 0):
            flist.sort()
        else:
            var_dir = self.log_dir / var_name
            if(var_dir.is_dir()):
                flist = list(var_dir.glob('*.*'))
            if(len(flist) > 0):
                flist.sort()
            else:
                self.logger('No such variable')
                return
        var_path = flist[single_shot_index]
                
        if(var_path.is_file()):
            self.logger(f'Loading {var_path}')
            if(var_path.suffix == '.npz'):
                buf = np.load(var_path)
                try:
                    time_array = buf['time_array']
                    n_logs = (time_array > 0).sum()
                    time_array = time_array[:n_logs]
                    data_array = buf['data_array']
                    data_array = data_array[:n_logs]
                    return((time_array, data_array))
                except:
                    return(buf)
            if(var_path.suffix == '.npy'):
                return(np.load(var_path))
            if(var_path.suffix == '.mat'):
                from scipy.io import loadmat
                return(loadmat(var_path))
            if(var_path.suffix == '.txt'):
                with open(var_path) as f_txt:
                    return(f_txt.read())
            if((var_path.suffix == '.tif') | (var_path.suffix == '.tiff')):
                from tifffile import imread
                return(imread(var_path))
            if(var_path.suffix == '.torch'):      
                from torch import load as torch_load 
                return(torch_load(var_path))
            try:
                from matplotlib.pyplot import imread
                img = imread(var_path)
                return(img)
            except:
                pass
        else:
            self.logger(f'{var_name} not found.')
            return
   
    def get_stack_of_files(self,
        var_name = None, flist = [], suffix = None,
        return_data = False, return_flist = True, read_func = None,
        data_makes_a_block = False, verbose = True):
       
        """ Get list or data of all files in a directory
       
            This function gives the list of paths of all files in a directory
            for a single variable.

            Parameters
            ----------
            :param var_name:
                The directory or variable name to look for the files
            :type var_name: str
           
            :param flist:
                list of Paths, if data is returned, this flist input can limit
                the data requested to this list.
            :type flist: list
           
            :param suffix:
                the suffix of files to look for, e.g. 'txt'
            :type siffix: str
           
            :param return_data:
                    with flist you can limit the data that is returned.
                    Otherwise the data for all files in the directory will be
                    returned
            :param return_flist:
                    Maybe you are only intrested in the flist.
                   
            :param data_makes_a_block:
                    if you know that the shape of all numpy arrays in files, or
                    images are the same, set this as true and receive a numpy
                    array. Otherwise returns a list. default: False
           
            Output
            ----------
           
                It returns a tuple, (dataset, flist),
                dataset will be a list of files contains or numpy array in
                case all files produce same shape numpy arrays.
                flist is type pathlib.Path
           
        """
        if suffix is None:
            var_name_split = var_name.split('.')
            if len(var_name_split) > 1:
                suffix = var_name_split[-1]
                name_before_suffix = var_name_split[:-1]
                if((len(name_before_suffix) == 1) &
                   (name_before_suffix[0] == '')):
                    var_name = '*'
                else:
                    var_name = ('.').join(var_name_split[:-1])
            else:
                suffix = '*'

        suffix = suffix.strip('.')
        if not flist:
            assert var_name is not None, \
                ' The file list is empty. Please provide the ' \
                + 'variable name or a non-empty file list.'
            var_dir = self.log_dir / var_name
            if(var_dir.is_dir()):
                var_fname = None
                flist = list(var_dir.glob(f'*.{suffix}'))
            else:
                var_fname = var_dir.name
                var_dir = var_dir.parent
                patt = f'{var_fname}*.{suffix}'
                while('**' in patt):
                    patt = patt.replace('**', '*')
                flist = list(var_dir.glob(patt))
        else:
            var_dir = flist[0].parent
            assert var_dir.is_dir(),\
                f'The directory {var_dir} for the '\
                + 'provided list cannot be accessed.'
           
        if flist:
            flist.sort()
            n_files = len(flist)
            if((not return_data) & return_flist):
                return(flist)
           
            ######### asked for data ########
            if(read_func is None):
                try:
                    fdata = np.load(flist[0])
                    read_func = np.load
                except:
                    pass
            if(read_func is None):
                try:
                    from matplotlib.pyplot import imread
                    fdata = imread(flist[0])
                    read_func = imread
                except:
                    pass
            if(read_func is not None):
                assert callable(read_func), \
                    f'given read_func: {read_func} is not callable.'
                fdata = read_func(flist[0])
                if(data_makes_a_block):
                    dataset = np.zeros((n_files, ) + fdata.shape,
                                       dtype=fdata.dtype)
                    if(verbose):
                        self.logger(f'logviewer: Reading dataset from {var_dir}'
                                    f', the shape would be: {dataset.shape}')
                    for fcnt, fpath in enumerate(flist):
                        dataset[fcnt] = read_func(fpath)
                else:
                    dataset = [read_func(fpath) for fpath in flist]
                if(return_flist):
                    return(dataset, flist)
                else:
                    return(dataset)
            else:
                if(verbose):
                    self.logger(f'File {flist[0].name} cannot be opened by '\
                          + r'np.load() or plt.imread(), provide read_func.')

    def get_common_files(self, var_name_A, var_name_B):
        """ get common files in two directories
        
            It happens often in ML that there are two directories, A and B,
            and we are interested to get the flist in both that is common 
            between them. returns a tuple of two lists of files.
            
            Parameters
            ----------
            :param var_name_A:
                directory A name
            :param var_name_B:
                directory B name
        """
        flist_A = self.get_stack_of_files(
            var_name_A, return_data = False, return_flist = True)
        flist_B = self.get_stack_of_files(
            var_name_B, return_data = False, return_flist = True)
        
        suffix_A = flist_A[0].suffix
        suffix_B = flist_B[0].suffix 
        parent_A = flist_A[0].parent
        parent_B = flist_B[0].parent
        
        fstems_A = [_fst.stem for _fst in flist_A]
        fstems_B = [_fst.stem for _fst in flist_B]
        
        fstems_A_set = set(fstems_A)
        fstems_B_set = set(fstems_B)
        common_stems = list(fstems_A_set.intersection(fstems_B_set))

        flist_A_new = [parent_A / (common_stem + suffix_A) \
                          for common_stem in common_stems]
        flist_B_new = [parent_B / (common_stem + suffix_B) \
                          for common_stem in common_stems]

        return(flist_A_new, flist_B_new)
    
    def __repr__(self):
        return f'{self.log_dir}'

    def __bool__(self):
        return self.log_dir.is_dir()

def str2type(_element):
    if _element[0] == '\'':
        return _element[1:-1]
    else:
        try:
            return int(_element)
        except ValueError:
            try:
                return float(_element)
            except ValueError:
                pass
    return _element

def text_to_object(txt):
    """ Read a list or dict that was sent to write to text e.g. via log_single:
    As you may have tried, it is possible to send a Pythonic list to a text file
    the list will be typed there with [ and ] and ' and ' for strings with ', '
    in between. In this function we will merely return the actual content
    of the original list.
    Now if the type the element of the list was string, it would put ' and ' in
    the text file. But if it is a number, no kind of punctuation or sign is 
    used. by write(). We support int or float. Otherwise the written text
    will be returned as string with any other wierd things attached to it.
    
    """
    if(txt[0] == '['):
        txt = txt.strip('[').strip(']')
        txt = txt.split(', ')
        obj_out = txt
        for cnt, _element in enumerate(txt):
            obj_out[cnt] = str2type(_element)
    elif(txt[0] == '{'):
        txt = txt.strip('{').strip('}')
        txt = txt.split(', ')
        obj_out = dict()
        for cnt, _element in enumerate(txt):
            _element_key = str2type(_element.split(': ')[0])
            _element_value = str2type(_element.split(': ')[1])
            obj_out[_element_key] = _element_value
    else:
        obj_out = txt
    return obj_out