from re import sub as re_sub
from unicodedata import normalize as unicodedata_normalize

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