from re import sub as re_sub
import numpy as np

def dummy_function(*args, **kwargs): ...

def is_builtin_collection(obj):
    """
    Determine if an object is a strictly built-in Python collection.
    
    This function uses a heuristic based on the object type's module being either 'builtins',
    'collections', or 'collections.abc', excluding strings and bytes explicitly, to identify
    if the given object is a built-in collection type (list, tuple, dict, set). It checks if the
    object belongs to one of Python's built-in collection modules and possesses both __len__ and
    __iter__ methods, which are typical characteristics of collections.
    
    Args:
        obj: The object to be checked.
    
    Returns:
        bool: True if the object is a built-in Python collection (excluding strings and bytes),
              False otherwise.
    
    Note:
        This function aims to exclude objects from external libraries (e.g., NumPy arrays) that,
        while iterable and having a __len__ method, are not considered built-in Python collections.
    """
    obj_type = type(obj)
    module = obj_type.__module__
    if ( (module not in ('builtins', 'collections', 'collections.abc'))
         | isinstance(obj, (str, bytes)) 
        ):
        return False
    return hasattr(obj, '__len__') and hasattr(obj, '__iter__')

def name_from_file(log_dir, fpath):
    """ 
        Given an fpath inside the logger log_dir, 
        what would be its equivalent parameter_name?
    """
    fpath_str = str(fpath.absolute())
    try:
        log_dir = str(log_dir.absolute())
    except:
        log_dir = str(log_dir)
    log_dir_str = None
    if log_dir in fpath_str:
        log_dir_str = log_dir
    if (log_dir + '/') in fpath_str:
        log_dir_str = log_dir + '/'
    if log_dir_str:
        fpath_name = fpath_str.split(log_dir_str)[-1]
        fpath_split = fpath_name.split('.')
        return '.'.join(fpath_split[:-1])
    
def repr_raw(text):
    """ Raw text representation
        Returns a raw string representation of a text that has escape 
        charachters
        
        Parameters:
        ^^^^^^^^^
        :param text:
        the input text, returns the fixed string
        
    """
    escape_dict={'\a':r'\a',
                 '\b':r'\b',
                 '\c':r'\c',
                 '\f':r'\f',
                 '\n':r'\n',
                 '\r':r'\r',
                 '\t':r'\t',
                 '\v':r'\v',
                 '\'':r'\'',
                 '\"':r'\"'}
    new_string=''
    for char in text:
        try: 
            new_string += escape_dict[char]
        except KeyError: 
            new_string += char
    return new_string

def replace_all(text, pattern, fill_value):
    """replace all instances of a pattern in a string with a new one
    """
    while (len(text.split(pattern)) > 1):
        text = text.replace(pattern, fill_value)
    return text

def select_directory(default_directory = './'):
    """ Open dialog to select a directory
        It works for windows and Linux using PyQt5.
    
       :param default_directory: pathlib.Path
                When dialog opens, it starts from this default directory.
    """
    from PyQt5.QtWidgets import QFileDialog, QApplication
    _ = QApplication([])
    log_dir = QFileDialog.getExistingDirectory(
        None, "Select a directory", default_directory, QFileDialog.ShowDirsOnly)
    return(log_dir)

def select_file():
    """ Open dialog to select a file
        It works for windows and Linux using PyQt5.
    """
    from PyQt5.QtWidgets import QFileDialog, QApplication
    from pathlib import Path
    _ = QApplication([])
    fpath = QFileDialog.getOpenFileName()
    fpath = Path(fpath[0])
    return(fpath)

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

def stack_to_frame(stack, frame_shape : tuple = None, borders = 0):
    """ turn a stack of images into a 2D frame of images
        This is very useful when lots of images need to be tiled
        against each other.
    
        Note: if the last dimension is 3, all images are RGB, if you don't wish that
        you have to add another dimension at the end by np.expand_dim(arr, axis = -1)
    
        :param stack: np.ndarray
                It must have the shape of either
                n_im x n_r x n_c
                n_im x n_r x  3  x  1
                n_im x n_r x n_c x  3
                
            In all cases n_im will be turned into a frame
            Remember if you have N images to put into a square, the input
            shape should be 1 x n_r x n_c x N
        :param frame_shape: tuple
            The shape of the frame to put n_rows and n_colmnss of images
            close to each other to form a rectangle of image.
        :param borders: literal or np.inf or np.nan
            When plotting images with matplotlib.pyplot.imshow, there
            needs to be a border between them. This is the value for the 
            border elements.
            
        output
        ---------
            Since we have N channels to be laid into a square, the side
            length would be ceil(N**0.5) if frame_shape is not given.
            it produces an np.array of shape n_f x n_r * f_r x n_c * f_c or
            n_f x n_r * f_r x n_c * f_c x 3 in case of an RGB input.
    """
    is_rgb = stack.shape[-1] == 3
    
    if(len(stack.shape) == 4):
        if((stack.shape[2] == 3) & (stack.shape[3] == 1)):
            stack = stack[..., 0]
    
    n_im, n_R, n_C = stack.shape[:3]
        
    if(len(stack.shape) == 4):
        assert is_rgb, 'For a stack of images with axis 3, it should be 1 or 3.'

    assert (len(stack.shape) == 3) | (len(stack.shape) == 4), \
        f'The stack you provided can have specific shapes. it is {stack.shape}'

    if(frame_shape is None):
        square_side = int(np.ceil(np.sqrt(n_im)))
        frame_n_r, frame_n_c = (square_side, square_side)
    else:
        frame_n_r, frame_n_c = frame_shape
    n_R += 2
    n_C += 2
    new_n_R = n_R * frame_n_r
    new_n_C = n_C * frame_n_c

    if is_rgb:
        frame = np.zeros((new_n_R, new_n_C, 3), dtype = stack.dtype)
    else:
        frame = np.zeros((new_n_R, new_n_C), dtype = stack.dtype)
    used_ch_cnt = 0
    if(borders is not None):
        frame += borders
    for rcnt in range(frame_n_r):
        for ccnt in range(frame_n_c):
            ch_cnt = rcnt + frame_n_c*ccnt
            if (ch_cnt<n_im):
                frame[rcnt*n_R + 1: (rcnt + 1)*n_R - 1,
                      ccnt*n_C + 1: (ccnt + 1)*n_C - 1] = \
                    stack[used_ch_cnt]
                used_ch_cnt += 1
    return frame

def stacks_to_frames(stack_list, frame_shape : tuple = None, borders = 0):
    """ turn a list of stack of images into a list of frame of images
        This is simply a list of calling stack_to_frame
        :param stack_list:
            It must have the shape of either
            n_f x n_im x n_r x n_c
            n_f x n_im x n_r x  3  x 1
            n_f x n_im x n_r x n_c x 3

    """    
    return np.array([stack_to_frame(stack, 
                           frame_shape = frame_shape, 
                           borders = borders) for stack in stack_list])
	
class ssh_system:
	def __init__(self, hostname, username, password):
		import paramiko
		self.ssh_client = paramiko.SSHClient()
		self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
		self.ssh_client.connect(hostname = hostname, 
						   username = username,
						   password = password)
	
	def ssh_ls(self, path):
		stdin, stdout, stderr = self.ssh_client.exec_command(
            'ls ' + str(path))
		ls_result = stdout.readlines()
		return ls_result
		
	def ssh_scp(self, source, destination):
		...
        
