import pathlib
import numpy as np
from   matplotlib.pyplot import imread
from   .utils            import replace_all, dummy_function, name_from_file

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
        self.load = self.get_single
        self.log_dir_str = str(self.log_dir.absolute())
    
    def name_from_file(self, fpath):
        """ 
            Given an fpath inside the logger log_dir, 
            what would be its equivalent parameter_name?
        """
        return name_from_file(self.log_dir_str, fpath)
    
    def disable_logger(self):
        self.logger = dummy_function
    
    def get_flist(self, var_name, suffix = None):
        """ get list of files
            return the list of files for a saved variable.

            Parameters
            ----------
            :param var_name:
                variable name
            :param suffix:
                If there are different suffixes availble for a variable
                this input needs to be set. npy, npz, mat, and torch are
                supported.
        """
        var_name = var_name.replace('\t', '\\t').replace('\n', '\\n')\
            .replace('\r', '\\r').replace('\b', '\\b')

        flist = list((self.log_dir).glob(var_name))
        
        if not flist:
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
                    suffix = '*'
    
            suffix = suffix.strip('.')        
    
            flist = []            
            if((self.log_dir / var_name).is_file()):
                flist = [self.log_dir / var_name]
            elif((self.log_dir / f'{var_name}.{suffix}').is_file()):
                flist = [self.log_dir / f'{var_name}.{suffix}']
            else:
                _var_name = (self.log_dir / var_name).name
                _var_dir = (self.log_dir / var_name).parent
                search_patt = f'{_var_name}*.{suffix}'
                search_patt = replace_all(search_patt, '**', '*')
                flist = list(_var_dir.glob(search_patt))
                if(len(flist) == 0):
                    search_patt = f'{_var_name}*.*'
                    search_patt = replace_all(search_patt, '**', '*')
                    flist = list(_var_dir.glob(search_patt))
                    if(len(flist) > 0):
                        self.logger(
                            'I Can not find the file with the given suffix, '\
                            + 'but found some with a different suffix, '\
                            + f'one file is: {flist[-1]}. This is what I'\
                            + ' will return.' )
                    
        if(flist):
            flist.sort()
        else:
            var_dir = self.log_dir / var_name
            if(var_dir.is_dir()):
                flist = list(var_dir.glob('*'))
            if(len(flist) > 0):
                flist.sort()
        return flist

    def get_common_files(self, var_name_A, var_name_B, suffix = None,
                               flist_A = None, flist_B = None):
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
        if not flist_A:
            flist_A = self.get_flist(var_name_A, suffix)
        if not flist_B:
            flist_B = self.get_flist(var_name_B, suffix)
        
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

    # def load(self, parameter_name, suffix = None, flist = None,
    #              file_index : [int, list[int]] = -1, read_func = None):
    #     ...
    
    def get_text(self, log_name='main_log', flist = None, suffix = 'txt',
                       file_index = -1):
        """ get text log files
            Given the log_name, this function returns the text therein.

            Parameters
            ----------
            :param log_name:
                the log name. If not given then it is the main log.
            :param flist:
                you can give a file list in Posix paths, for text files
            :param suffix: str
                to search for specifi files
            :param file_index: int or list[int]
                a number or a list of numbers for the index of the file 
                to include, default: -1

        """
        if isinstance(file_index, int):
            file_index = [file_index]
        if not flist:
            flist = self.get_flist(log_name, suffix)
        n_files = len(flist)
        if (n_files>0):
            txt = []
            for fcnt in file_index:
                with open(flist[int(fcnt)]) as f_txt:
                    txt.append(f_txt.readlines())
            if(n_files == 1):
                txt = txt[0]
            return txt

    def get_single(self, var_name, file_index = -1, 
                   suffix = None, read_func = None, verbose = False):
        """ get a single variable
            return the value of a saved variable.

            Parameters
            ----------
            :param var_name:
                variable name
            :param file_index:
                If there are many snapshots of a variable, this input can
                limit the returned to a set of indices.
            :param suffix:
                If there are different suffixes availble for a variable
                this input needs to be set. npy, npz, mat, and torch are
                supported.
            :param read_func:
                a function that takes the Posix path and returns data
            .. note::
                when reading a MATLAB file, the output is a dictionary.
                Also when reading a npz except if it is made by log_var
        """
        assert file_index == int(file_index), \
                    f'file_index {file_index} must be an integer'
        flist = self.get_flist(var_name, suffix)
        var_path = None
        if flist:
            if len(flist)>1:
                if verbose:
                    self.logger(  f'There are {len(flist)} files, logged with'
                                + f' name {var_name}.'
                                + f' The given index is {file_index}.')
            var_path = flist[file_index]
    
            if(var_path.is_file()):
                if verbose:
                    self.logger(f'Loading {var_path}')
                if read_func is not None:
                    return read_func(var_path)
                if(var_path.suffix == '.npz'):
                    buf = np.load(var_path)
                    try: #check if it is made by log_var
                        assert len(buf.files) == 2
                        time_array = buf['time_array']
                        data_array = buf['data_array']
                        data_array = data_array[time_array > 0]
                        time_array = time_array[time_array > 0]
                        return((time_array, data_array))
                    except:
                        return(buf)
                if(var_path.suffix == '.npy'):
                    return(np.load(var_path))
                if(var_path.suffix == '.mat'):
                    from scipy.io import loadmat
                    return(loadmat(var_path))
                if(var_path.suffix == '.dm4'):
                    from hyperspy.api import load as hyperspy_api_load
                    return hyperspy_api_load(var_path).data
                if((var_path.suffix == '.tif') | (var_path.suffix == '.tiff')):
                    from tifffile import imread
                    return(imread(var_path))
                if(var_path.suffix == '.torch'):      
                    from torch import load as torch_load 
                    return(torch_load(var_path))
                try:
                    img = imread(var_path)
                    return(img)
                except:
                    pass
                # if( (var_path.suffix in ['.txt', '.pdb', '.json', '.fasta'])):
                #     return var_path.read_text()
                return var_path.read_text()
            else:
                var_path = None
                
        if (var_path is None) & verbose:
            self.logger(f'Looking for {var_name} failed. ' + \
                        f'{var_path} is not in: {self.log_dir}')
    
    def get_stack_of_files(self, 
        var_name = None, flist = [], suffix = None, read_func = None):
       
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
           
            :param read_func:
                the function that takes the posix path of a file and returns
                the data in there.
           
            Output
            ----------
           
                It returns a list of data in all files or a numpy array if 
                concatenation of all is possible.
        """
        if not flist:
            flist = self.get_flist(var_name, suffix)
        else:
            flist = list(flist)
            assert pathlib.Path(flist[0]).is_file(), \
                f'File not found: {flist[0]}. You can use logviewer get_flist'
        
        if flist:
            n_files = len(flist)
            if(read_func is None):
                try:
                    fdata = np.load(flist[0])
                    read_func = np.load
                except:
                    pass
            if(read_func is None):
                try:
                    fdata = imread(flist[0])
                    read_func = imread
                except:
                    pass
            try:
                read_func(flist[0])
            except e:
                self.logger(f'The data file {flist[0]} could not be opened.'
                            'Please provide a read_function in the input.')
                raise e
            dataset = [read_func(fpath) for fpath in flist]
            try:
                dataset = np.array(dataset)
            except:
                pass
            return(dataset)

    def replace_time_with_index(self, var_name):
        """ index in file names
            lognflow uses time stamps to make new log files for a variable.
            That is done by putting time stamp after the name of the variable.
            This function changes all of the time stamps, sorted ascendingly,
            by indices.
            
            :param var_name:
                variable name
        """
        var_dir = self.log_dir / var_name
        if(var_dir.is_dir()):
            var_fname = None
            flist = list(var_dir.glob(f'*.*'))
        else:
            var_fname = var_dir.name
            var_dir = var_dir.parent
            flist = list(var_dir.glob(f'{var_fname}_*.*'))
        if flist:
            flist.sort()
            fcnt_width = len(str(len(flist)))
            for fcnt, fpath in enumerate(flist):
                # self.log_text(None, f'Changing {flist[fcnt].name}')
                fname_new = ''
                if(var_fname is not None):
                    fname_new = var_fname + '_'
                fname_new += f'{fcnt:0{fcnt_width}d}' + flist[fcnt].suffix
                fpath_new = flist[fcnt].parent / fname_new
                # self.log_text(None, f'To {fpath_new.name}')
                flist[fcnt].rename(fpath_new)
        
    def __repr__(self):
        return f'{self.log_dir}'

    def __bool__(self):
        return self.log_dir.is_dir()