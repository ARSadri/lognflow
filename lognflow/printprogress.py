from math import ceil
from time import time

class printprogress:
    """
    While there are modules that use \r to show a progress bar, 
    there are cases e.g. a print_function or an ssh terminals that do not 
    support \r. In such cases, if something is written at the end of 
    the line, it cannot be deleted. The following code provides a good 
    looking simple title for the progress bar and shows the limits and 
    is very simple to use. Define the object with number of steps of the loop 
    and then  call it in every iteration of the loop.
    """
    def __init__(self, 
                 n_steps, 
                 numTicks = 78,
                 title = None,
                 print_function = print):
        """
            n_steps: int
                usually the number of iterations in the for loop
            numTicks: int
                The number of charachters in a row of the screen - 2
                default: 78
            title : str 
                The title of progress bar.
                default: f'Progress bar for {n_steps} steps in {numTicks} ticks'
            print_function:
                print_function must be callable with a string and should not add
                \n to the its input just because its been called.
        """
        
        self.in_print_function = print_function
        
        if(n_steps != int(n_steps)):
            self._print_function('textProgBar takes integers no less than 2 ')
            n_steps = int(n_steps)
        if(n_steps<2):
            n_steps = 2
        if (title is None):
            title = f'Progress bar for {n_steps} steps in {numTicks} ticks'
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
        for _ in range(self.numTicks):
            self._print_function('_', end='')
        self._print_function(' ')
        self._print_function('/', end='')
        for idx in range(self.numTicks - len(title)):
            if(idx==int((self.numTicks - len(title))/2)):
                self._print_function(title, end='')
            else:
                self._print_function(' ', end='')
        self._print_function(' \\')
        self._print_function(' ', end = '')
    
    def _print_function(self, text, end='\n'):
        if (self.in_print_function == print):
            print(text, end = end, flush = True)
        else:
            self.in_print_function(text)
        
    def __call__(self, ck=1):
        """ ticking the progress bar
            just call the object and the progress bar moves one agead.
                   p                   x      
            |-----------|--------------------------|
                       ck                       n_steps
            As such  x/(n-c) = p/c => x = p(n/c - 1)
        """
        if(self.FLAG_ended):
            if(not self.FLAG_warning):
                self.FLAG_warning = True
                self._print_function('-' * self.numTicks)
        else:
            self.ck += ck
            if(self.ck >= self.n_steps):
                self.end()
            else:
                cProg = int(self.numTicks*self.ck/(self.n_steps-1)/3)    
                #3: because 3 charachters are used
                while((self.prog < cProg) & (not self.FLAG_ended)):
                    self.prog += 1
                    passedTime = time() - self.startTime
                    remTimeS = passedTime * ( self.n_steps / self.ck - 1)
                    if(remTimeS>=5940):
                        progStr = "%02d" % int(ceil(remTimeS/3600))
                        self._print_function(progStr, end='')
                        self._print_function('h', end='')
                    elif(remTimeS>=99):
                        progStr = "%02d" % int(ceil(remTimeS/60))
                        self._print_function(progStr, end='')
                        self._print_function('m', end='')
                    elif(remTimeS>0):
                        progStr = "%02d" % int(ceil(remTimeS))
                        self._print_function(progStr, end='')
                        self._print_function('s', end='')
                    else:
                        self.end()
    def end(self):
        if(not self.FLAG_ended):
            self._print_function('')
            self.FLAG_ended = True