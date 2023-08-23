from math import ceil
from time import time

class printprogress:
    """
    While there are packages that use \r to show a progress bar, 
    there are cases e.g. a print_function or an ssh terminal that does not 
    support \r. In such cases, if something is typed at the end of 
    the line, it cannot be deleted. The following code provides a good 
    looking progress bar with a simple title and limits and 
    is very simple to use. Define the object with number of steps of the loop 
    and then call it in every iteration of the loop. If you'd like it
    to go faster, call it with give number of steps that have passed.
    """
    def __init__(self, 
                 n_steps, 
                 numTicks = 78,
                 title = None,
                 method = 'linear',
                 print_function = print,
                 **print_function_kwargs):
        """
            n_steps: int
                Number of iterations in the for loop
            numTicks: int
                The number of charachters in a row of the screen - 2
                default: 78 for old screens that have 80 coloumns
            title : str 
                The title of progress bar.
                default: f'Progress bar for {n_steps} steps in {numTicks} ticks'
            print_function:
                print_function must be callable with a string and should not add
                \n to the its input.
                If you pass None as the print_function, nothing will be printed.
                yet the output of the __call__ will be the remaining time in
                seconds.
            method: options how to calculate the remaining time are
                'linear':
                       p                   x      
                |-----------|--------------------------|
                           ck                       n_steps
                As such  x/(n-c) = p/c => x = p(n/c - 1)
                more options to come
        """
        assert method in ['linear', 'linear_robust']
        self.print_function_kwargs = print_function_kwargs
        self.method = method
        self.in_print_function = print_function
        if(n_steps != int(n_steps)):
            self._print_func(
                r'printprogress takes integers no less than 2 as n_steps.')
            n_steps = int(n_steps)
        if(n_steps<2):
            n_steps = 2
        if (title is None):
            title = f'Progress for {n_steps} steps'
        self.FLAG_ended = False
        self.FLAG_warning = False
        self.startTime = time()
        self.ck = 0
        self.prog = 0
        self.n_steps = n_steps
        if(numTicks < len(title) + 2 ):
            self.numTicks = len(title)+2
        else:
            self.numTicks = numTicks
        
        self._print_func(' ', end='')
        self._print_func('_'*self.numTicks, end='')
        self._print_func(' ')
        
        self._print_func('/', end='')
        self._print_func(' '*int((self.numTicks - len(title))/2), end='')
        self._print_func(title, end='')
        self._print_func(' '*int(ceil((self.numTicks-len(title))/2)-1), end='')
        self._print_func(' \\')
        
        self._print_func(' ', end = '')
        self.len_prog_text = 0
    
    def _print_func(self, text, end='\n'):
        if (self.in_print_function is not None):
            if (self.in_print_function == print):
                print(text, end = end, flush = True)
            else:
                self.in_print_function(text, end = end,
                                       **self.print_function_kwargs)
        
    def _calc_ETA(self):
        if(self.method == 'linear'):
            passedTime = time() - self.startTime
            remTimeS = passedTime * ( self.n_steps / self.ck - 1)
        
        return remTimeS
    
    def __call__(self, ck=1):
        """ ticking the progress bar
            just call the object and the progress bar moves ck steps
            ahead when ready.
            
            output
            ~~~~~~
            :param ETA:
                the remaining time in seconds will be provided at the output
        """
        remTimeS = 0
        if(self.FLAG_ended):
            if(not self.FLAG_warning):
                self.FLAG_warning = True
                self._print_func('-' * (self.numTicks + 2))
        else:
            self.ck += ck
            if(self.ck <= self.n_steps):
                remTimeS = self._calc_ETA() # useful when print_function is None
                cProg = int(self.numTicks*self.ck/(self.n_steps-1)/3)
                #3: because 3 charachters are used
                while((self.prog < cProg) & (not self.FLAG_ended)):
                    self.prog += 1
                    remTimeS = self._calc_ETA()
                    if(remTimeS>356400): # less than 99d and more than 99h
                        progStr = "%02d" % int(ceil(remTimeS/86400))
                        self._print_func(progStr, end='')
                        self._print_func('d', end='')
                        self.len_prog_text += 3
                    elif(remTimeS>5940): # less than 99h and more than 99m
                        progStr = "%02d" % int(ceil(remTimeS/3600))
                        self._print_func(progStr, end='')
                        self._print_func('h', end='')
                        self.len_prog_text += 3
                    elif(remTimeS>99): # less than 99m and more than 99s
                        progStr = "%02d" % int(ceil(remTimeS/60))
                        self._print_func(progStr, end='')
                        self._print_func('m', end='')
                        self.len_prog_text += 3
                    elif(remTimeS>=0): # less than 99s and more than 0
                        progStr = "%02d" % int(ceil(remTimeS))
                        self._print_func(progStr, end='')
                        self._print_func('s', end='')
                        self.len_prog_text += 3
                    else:
                        self._end()
            if((self.ck >= self.n_steps) | 
               (self.len_prog_text >= self.numTicks)):
                self._end()
        return remTimeS

    def _end(self):
        if(not self.FLAG_ended):
            self._print_func('')
            self.FLAG_ended = True
            
    def __del__(self):
        self._end()