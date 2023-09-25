from re import sub as re_sub
import numpy as np
from unicodedata import normalize as unicodedata_normalize

def dummy_function(*args, **kwargs): ...

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

def multichannel_to_frame(stack, frame_shape : tuple = None, borders = 0):
    """ turn a stack of multi-channel images into a frame of images
        This is very useful when lots of images need to be tiled
        against each other.
    
        :param stack: np.ndarray
                It must have the shape of either
                n_r x n_c x n_ch
                n_r x n_c x  3  x n_ch
                n_f x n_r x n_c x n_ch
                n_f x n_r x n_c x  3  x n_ch
                
            In both cases n_ch will be turned into a square tile
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
            length woul be ceil(N**0.5)
            it produces an np.array of shape n_f x n_r * S x n_c * S or
            n_f x n_r * S x n_c * S x 3 in case of RGB input.
    """
    if(len(stack.shape) == 4):
        if(stack.shape[3] == 3):
            stack = np.array([stack])
    if(len(stack.shape) == 3):
        stack = np.array([stack])
    
    if((len(stack.shape) == 4) | (len(stack.shape) == 5)):
        if(len(stack.shape) == 4):
            n_imgs, n_R, n_C, n_ch = stack.shape
        if(len(stack.shape) == 5):
            n_imgs, n_R, n_C, is_rgb, n_ch = stack.shape
            if(is_rgb != 3):
                return None
        if(frame_shape is None):
            square_side = int(np.ceil(np.sqrt(n_ch)))
            frame_n_r, frame_n_c = (square_side, square_side)
        else:
            frame_n_r, frame_n_c = frame_shape
        
        new_n_R = n_R * frame_n_r
        new_n_C = n_C * frame_n_c
        if(len(stack.shape) == 4):
            canv = np.zeros((n_imgs, new_n_R, new_n_C), 
                            dtype = stack.dtype)
        if(len(stack.shape) == 5):
            canv = np.zeros((n_imgs, new_n_R, new_n_C, 3),
                             dtype = stack.dtype)
        used_ch_cnt = 0
        if(borders is not None):
            stack[:,   :1      ] = borders
            stack[:,   : ,   :1] = borders
            stack[:, -1:       ] = borders
            stack[:,   : , -1: ] = borders
        
        for rcnt in range(frame_n_r):
            for ccnt in range(frame_n_c):
                ch_cnt = rcnt + frame_n_c*ccnt
                if (ch_cnt<n_ch):
                    canv[:, rcnt*n_R: (rcnt + 1)*n_R,
                            ccnt*n_C: (ccnt + 1)*n_C] = \
                        stack[..., used_ch_cnt]
                    used_ch_cnt += 1
    else:
        return None
    return canv
	
class ssh_stablish:
	def __init__(self, hostname, username, password):
		import paramiko

		ssh_client = paramiko.SSHClient()
		ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
		ssh_client.connect(hostname = hostname, 
						   username = username,
						   password = password)
		self.ssh_client = ssh_client
	
	def ssh_ls(self, ssh_client, path):
		stdin, stdout, stderr = ssh_client.exec_command('ls ' + results_path)
		ls_result = stdout.readlines()
		return ls_result
		
	def ssh_scp(self, ssh_client, source, destination):
		...