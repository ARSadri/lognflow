from multiprocessing import Process, Queue, cpu_count, Event

def _loopprocessor_function_test_mode(
         targetFunction, theQ, procID_range, error_event, args, kwargs):
    results = targetFunction(*args, **kwargs)
    theQ.put([procID_range, [results], False])

def _loopprocessor_function(
        targetFunction, theQ, procID_range, error_event, args, kwargs):
    try:
        result = targetFunction(*args, **kwargs)
        theQ.put([procID_range, [result], False])
    except Exception as e:
        if not error_event.is_set():
            error_event.set()
        theQ.put([procID_range, None, True])

class loopprocessor():
    def __init__(self, 
            targetFunction, n_cpu = None, test_mode = False, logger = print,
            verbose = True):
        self.targetFunction = targetFunction
        self.test_mode = test_mode
        self.aQ = Queue()

        if(n_cpu is None):
            self.n_cpu = cpu_count()
        else:
            self.n_cpu = n_cpu
        self.verbose = verbose
        if(self.verbose):
            self.logger = logger
            self.logger(f'lognflow loopprocessor initialized with {self.n_cpu} CPUs.')

        self.outputs_is_given = False
        self.outputs = []
        self.Q_procID = []
    
        self.numBusyCores = 0
        self.procID = 0
        self.numProcessed = 0

        self.any_error = False
        self.error_event = Event()
        self.empty_queue = False
    
    def __call__(self, *args, **kwargs):
        if (len(args) == 0) & (len(kwargs) == 0):
            self.empty_queue = True
        
        release_a_cpu = False
        if (len(args) > 0) | (len(kwargs) > 0):
            if(self.numBusyCores >= self.n_cpu):
                release_a_cpu = True
                
        single_queue_access = True
        while(single_queue_access | release_a_cpu | self.empty_queue):
            single_queue_access = False
        
            if (not self.aQ.empty()):
                aQElement = self.aQ.get()
                ret_procID_range = aQElement[0]
                ret_result = aQElement[1]
                if ((not self.any_error) & aQElement[2]):
                    self.any_error = True
                    self.empty_queue = True
                    error_ret_procID = ret_procID_range.copy()
                    self.logger('')
                    self.logger('lognflow, loopprocessor: An exception'\
                                ' has been raised. signaling all processes'\
                                ' to stop and join, please wait...')
                if (not self.any_error):
                    for ret_procID, result in zip(ret_procID_range, ret_result):
                        self.Q_procID.append(ret_procID)
                        self.outputs.append(result)
                elif(self.numBusyCores):
                    self.logger(f'Number of busy cores: {self.numBusyCores}')
    
                self.numProcessed += 1
                self.numBusyCores -= 1
                release_a_cpu = False
                if(self.any_error & (self.numBusyCores == 0)):
                    self.logger(f'Number of busy cores: {self.numBusyCores}')
                    self.logger(f'All cores are free')
                    self.empty_queue = False
                    break
            if(self.numProcessed >= self.procID):
                self.empty_queue = False
                
        if(not self.any_error):
            if (len(args) > 0) | (len(kwargs) > 0):
                procID_range = [self.procID]
                _args = (
                    self.targetFunction, self.aQ, procID_range, 
                    self.error_event, args, kwargs)
                if(self.test_mode):
                    _loopprocessor_function_test_mode(*_args)
                else:
                    Process(target = _loopprocessor_function, 
                            args = _args).start()
                self.procID += 1
                self.numBusyCores += 1
    
        if(self.any_error):
            self.logger('-'*79)
            self.logger('An exception occured during submitting jobs.')
            self.logger('Here we try to reproduce it but will raise '
                  'ChildProcessError regardless.')
            self.logger(f'We will call {self.targetFunction} ')
            self.logger('with the following index to slice the inputs:'
                  f' {error_ret_procID[0]}')
            self.logger('to avoid seeing this message, pass the argument called '\
                   'legger, it is print by default.')
            self.logger('-'*79)
            _loopprocessor_function_test_mode(
                self.targetFunction, self.aQ, 
                error_ret_procID, self.error_event, args, kwargs)
            raise ChildProcessError
        
        if (len(args) == 0) & (len(kwargs) == 0):
            return self.outputs