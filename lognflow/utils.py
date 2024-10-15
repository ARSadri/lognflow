import re
import time
import inspect
import pathlib
import numpy as np
from pathlib import Path

def dummy_function(*args, **kwargs): ...

class DummyClass:
    def __init__(self, *args, **kwargs):
        print('warning: this is a dummy class! It returns '
              'None to whatever you throw at it.')

    def __getattr__(self, name):
        return self.dummy_function

    def __call__(self, *args, **kwargs):
        return None

    def dummy_function(self, *args, **kwargs):
        return None

def is_builtin_collection(obj):
    """
    Determine if an object is a strictly built-in Python collection.
    
    This function uses a heuristic based on the object 
    type's module being either 'builtins',
    'collections', or 'collections.abc', excluding 
    strings and bytes explicitly, to identify
    if the given object is a built-in collection 
    type (list, tuple, dict, set). It checks if the
    object belongs to one of Python's built-in 
    collection modules and possesses both __len__ and
    __iter__ methods, which are typical characteristics of collections.
    
    Args:
        obj: The object to be checked.
    
    Returns:
        bool: True if the object is a built-in Python 
        collection (excluding strings and bytes),
              False otherwise.
    
    Note:
        This function aims to exclude objects from external 
        libraries (e.g., NumPy arrays) that,
        while iterable and having a __len__ method, 
        are not considered built-in Python collections.
    """
    obj_type = type(obj)
    module = obj_type.__module__
    if ( (module not in ('builtins', 'collections', 'collections.abc'))
         | isinstance(obj, (str, bytes)) 
        ):
        return False
    return hasattr(obj, '__len__') and hasattr(obj, '__iter__')

def assure_is_collection(returned_obj):
    if not is_builtin_collection(returned_obj):
        return [returned_obj]
    return returned_obj

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
    _ = QApplication([])
    fpath = QFileDialog.getOpenFileName()
    fpath = Path(fpath[0])
    return(fpath)

def text_to_collection(text):
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
    import ast
    def parse_node(node):
        if isinstance(node, ast.List):
            return [parse_node(elem) for elem in node.elts]
        elif isinstance(node, ast.Dict):
            return {parse_node(key): parse_node(value) 
                    for key, value in zip(node.keys, node.values)}
        elif isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Num):  # For Python < 3.8
            return node.n
        elif isinstance(node, ast.Str):  # For Python < 3.8
            return node.s
        elif isinstance(node, ast.Name):
            if node.id == 'array':
                return np
            elif node.id == 'tensor':
                import torch
                return torch
        elif isinstance(node, ast.Call):
            func_name = node.func.id
            if func_name == 'array':
                return np.array([parse_node(arg) for arg in node.args])
            elif func_name == 'tensor':
                import torch
                return torch.tensor([parse_node(arg) for arg in node.args])
        return None

    tree = ast.parse(text, mode='eval')
    return parse_node(tree.body)

class SSHSystem:
    """
    A class to handle basic SSH and SFTP operations on a remote system.

    Attributes:
        ssh_client (paramiko.SSHClient): The SSH client for executing 
        commands on the remote system.
        sftp_client (paramiko.SFTPClient): The SFTP client for file 
        transfer operations.
    """

    def __init__(self, hostname: str, username: str, password: str):
        """
        Initialize the SSHSystem by setting up the SSH and SFTP clients.

        Args:
            hostname (str): The hostname or IP address of the remote system.
            username (str): The username for SSH authentication.
            password (str): The password for SSH authentication.
        """
        import paramiko
        self.ssh_client = paramiko.SSHClient()
        self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            self.ssh_client.connect(
                hostname=hostname, username=username, password=password)
            self.sftp_client = self.ssh_client.open_sftp()
        except Exception as e:
            print(f"Failed to connect to {hostname}: {e}")
            self.ssh_client = None
            self.sftp_client = None

    def ssh_ls(self, path: Path):
        """
        List the contents of a directory on the remote system.

        Args:
            path (Path): The path to the directory on the remote system.

        Returns:
            list: A list of Path objects representing the files in the directory.
        """
        try:
            stdin, stdout, stderr = self.ssh_client.exec_command(f'ls {path}')
            ls_result = stdout.readlines()
            return [path / file.strip() for file in ls_result]
        except Exception as e:
            print(f"Error listing directory {path}: {e}")
            return []

    def ssh_scp(self, source: Path, destination: Path):
        """
        Copy a file from the remote system to the local system using SFTP.

        Args:
            source (Path): The path of the file on the remote system.
            destination (Path): The path where the file will be saved locally.
        """
        try:
            self.sftp_client.get(str(source), str(destination))
        except Exception as e:
            print(f"Error copying {source} to {destination}: {e}")

    def ssh_rm(self, path: Path):
        """
        Remove a file from the remote system.

        Args:
            path (Path): The path to the file to be removed.

        Returns:
            tuple: A tuple containing the stdout and stderr outputs from the command.
        """
        try:
            stdin, stdout, stderr = self.ssh_client.exec_command(f'rm {path}')
            return stdout.read().decode(), stderr.read().decode()
        except Exception as e:
            print(f"Error removing file {path}: {e}")
            return "", str(e)

    def monitor_and_remove(
        self, remote_folder: Path, local_folder: Path, 
        target_fname: str, interval=30
    ):
        """
        Monitor a remote folder for a specific file. Once the file appears,
        transfer and delete other files from the folder.

        Args:
            remote_folder (Path): The folder on the remote system to monitor.
            local_folder (Path): The local folder where files will be copied.
            target_fname (str): The name of the file to wait for.
            interval (int, optional): The time interval (in seconds) 
            between each check. Default is 30 seconds.
        """
        interesting_file_path = remote_folder / target_fname
        cnt = 0
        while not self.is_file(interesting_file_path):
            if (cnt % 100) == 0:
                print(f'Waiting for {interesting_file_path}', end='')
            else:
                print('.', end='', flush=True)
            time.sleep(interval)
            cnt += 1
        print('')

        print(f"{target_fname} found! Starting file transfer and deletion.")
        files = self.ssh_ls(remote_folder)
        for file in files:
            local_file = local_folder / file.name
            
            # Copy file to local folder
            print(f"Copying {file} to {local_file}")
            self.ssh_scp(file, local_file)
            
            # Delete file from remote server
            print(f"Deleting {file} from remote server")
            self.ssh_rm(file)

    def is_file(self, path: Path) -> bool:
        """
        Check if a file exists on the remote system.

        Args:
            path (Path): The path to the file on the remote system.

        Returns:
            bool: True if the file exists, False otherwise.
        """
        try:
            stdin, stdout, stderr = self.ssh_client.exec_command(
                f'test -f {path} && echo "exists"')
            return "exists" in stdout.read().decode()
        except Exception as e:
            print(f"Error checking file {path}: {e}")
            return False

    def close_connection(self):
        """
        Close the SSH and SFTP connections to the remote system.
        """
        if self.sftp_client:
            self.sftp_client.close()
        if self.ssh_client:
            self.ssh_client.close()

def printv(var, **kwargs):
    # Get the name of the variable passed to the function
    frame = inspect.currentframe().f_back
    var_name = [name for name, value in frame.f_locals.items() if value is var]
    
    # Ensure that var_name is not empty
    if var_name:
        var_name = var_name[0]
    else:
        var_name = 'variable'

    is_np_torch = True
    var_class = type(var).__name__
    toprint = f'{var_class} {var_name}: '
    try:
        array_shape = var.shape
        toprint += f'shape={array_shape}'
    except: 
        is_np_torch = False
    try:
        array_dtype = var.dtype
        toprint += f', dtype={array_dtype}'
    except: pass
    try:
        toprint += f', device={var.device}'
    except: pass
    
    if is_np_torch:
        arr_size = np.prod(array_shape)
        if 'array_size_threshold' in kwargs:
            array_size_threshold = kwargs['array_size_threshold']
        else:
            array_size_threshold = 1e+6
        if arr_size < array_size_threshold:
            try:
                toprint += f', min={var.min():.6f}'
            except: pass
            try:
                toprint += f', max={var.max():.6f}'
            except: pass
            try:
                toprint += f', mean={var.mean():.6f}'
            except: pass
            try:
                toprint += f', std={var.std():.6f}'
            except: pass
            
    if not is_np_torch:
        toprint += str(var)
    # Print the information
    print(toprint)

class Pyrunner:
    """
    A Jupyter-like Python code runner that executes code in blocks based 
    on cell numbers, supports saving and loading kernel states, and allows 
    interactive execution.

    Attributes:
        fpath (Path): The path to the Python file to execute.
        logger_ (callable): An optional logger function to log messages.
        log (str): A string containing the accumulated log messages.
        saved_state (dict): A dictionary to hold saved kernel states.
        exit (bool): A flag to indicate when to stop execution.
    """

    def __init__(self, fpath: str, logger=None, 
                 block_identifier = 'code_block_id'):
        """
        Initializes the Pyrunner class, runs the Python file in an interactive loop,
        and allows execution of specific code blocks identified by cell numbers.

        Args:
            fpath (str): The file path to the Python script to be executed.
            logger (callable, optional): A logger function to log output 
            (default is None).
        """
        self.block_identifier = block_identifier
        self.logger_ = logger
        self.fpath = Path(fpath)
        assert self.fpath.is_file(), f"File {fpath} does not exist."
        self.log = ''
        self.saved_state = {}
        self.exit = False

        self.logger(f'file: {fpath}')
        while not self.exit:        
            show_and_ask_result = self.show(globals())
            if show_and_ask_result is None:
                continue
            globals().update(show_and_ask_result)
            exec(globals().get('pyrunner_code', ''), globals())

    def logger(self, toprint: str, end: str = '\n'):
        """
        Logs the provided message. If a logger is provided, it logs the message 
        using that function. Otherwise, it appends the message to the internal log.

        Args:
            toprint (str): The message to log.
            end (str, optional): The string appended after each message 
            (default is '\n').
        """
        toprint = str(toprint) + end
        self.log += toprint
        if self.logger_ is not None:
            self.logger_(toprint)

    def save_or_load_kernel_state(self, globals_: dict, saved_state=None):
        """
        Saves or loads the kernel state using the `dill` library. 
        If `saved_state` is provided, it loads
        the state into `globals_`. If `saved_state` is None, it returns 
        a serialized form of the current global variables.

        Args:
            globals_ (dict): The global variables to save or update.
            saved_state (bytes, optional): The serialized kernel state to load 
            (default is None).

        Returns:
            bytes: A serialized version of the global variables if saving the state.
        """
        import dill as pickle
        if saved_state is None:
            return pickle.dumps(
                {k: v for k, v in globals_.items() 
                 if not k.startswith('__') and not callable(v)}
            )
        else:
            globals_.update(pickle.loads(saved_state))

    @property
    def n_saves(self) -> int:
        """
        Returns the number of saved states.

        Returns:
            int: The number of saved states.
        """
        return len(self.saved_state.keys())

    def show(self, globals_: dict, figsize: tuple = (3, 2)) -> dict:
        """
        Displays available cell blocks for execution and handles user 
        interaction to run specific blocks or manage kernel states 
        (save/load/delete).

        Args:
            globals_ (dict): The global variables of the current session.
            figsize (tuple, optional): The size of the dialog box 
            (default is (3, 2)).

        Returns:
            dict: A dictionary containing the updated global variables 
            if a cell block is selected.
        """
        pyrunner_code = open(self.fpath).read()
        pattern = r"if\s+" + self.block_identifier + "\s*==\s*(\d+):"
        matches = re.findall(pattern, pyrunner_code)

        if len(matches) == 0:
            self.logger(f'Running the pyrunner_code in {self.fpath}')
            self.logger(f'No code blocks found that checks {self.block_identifier}')
            return

        block_identifiers = sorted(set(int(num) for num in matches))
        buttons = {}

        for block_identifier in block_identifiers:
            buttons[f'{block_identifier}'] = block_identifier
        
        # Add options for saved states
        for key in self.saved_state:
            buttons[f'load_{key}'] = f'load_{key}'
            buttons[f'del_{key}'] = f'del_{key}'

        buttons[f'save_{self.n_saves + 1}'] = f'save_{self.n_saves + 1}'
        buttons['exit'] = 'exit'

        # Display dialog for user interaction
        from lognflow.plt_utils import question_dialog
        show_and_ask_result = question_dialog(
            question='Choose a cell number', figsize=figsize, buttons=buttons
        )
        if show_and_ask_result is None:
            self.logger(f'pyrunner: closing reloads, press Exit to close.')
            return

        # Handle user selection
        if isinstance(show_and_ask_result, str):
            if show_and_ask_result == 'exit':
                self.exit = True
                return

            elif 'save' in show_and_ask_result:
                key = show_and_ask_result.split('save_')[1]
                self.saved_state[key] = self.save_or_load_kernel_state(globals_)
                self.logger(f'Saved state: {key}')
                return

            elif 'load' in show_and_ask_result:
                key = show_and_ask_result.split('load_')[1]
                self.save_or_load_kernel_state(globals_, self.saved_state[key])
                self.logger(f'Loaded state: {key}')
                return

            elif 'del' in show_and_ask_result:
                key = show_and_ask_result.split('del_')[1]
                self.saved_state.pop(key)
                self.logger(f'Deleted state: {key}')
                return

        elif isinstance(show_and_ask_result, int):
            globals_['pyrunner_code'] = pyrunner_code
            globals_[self.block_identifier] = show_and_ask_result
            return globals_

