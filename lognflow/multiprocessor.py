#from-import is necessary to speed up spawning in Windows as much as possible
from numpy import __name__    as np___name__
from numpy import array       as np_array
from numpy import ceil        as np_ceil
from numpy import arange      as np_arange
from numpy import zeros       as np_zeros
from numpy import minimum     as np_minimum
from numpy import concatenate as np_concatenate

from multiprocessing import Process, Queue, cpu_count

from .printprogress import printprogress

class multiprocessor:
    """ multiprocessor makes the use of multiprocessing in Python easy
    
    Copyright: it was developed as part of the RobustGaussianFittingLibrary,
    however, since that library is not really about flow of algorithms and
    this one is, I moved it here.
    
    You would like to have a function that runs a process on a single entry and
    produces an output, then tell it to do the same thing on many entries.
    right?
    
    This is for you. Notice that your function should take an index (a single 
    integer) to refer to one of the enteries.
    
    We will produce many parallel processes and loop over all indices. We pass
    the index and the inputs (and if you allow, parts of inputs according to
    each index) to the function. Then collect all outputs and append them or 
    concatenate them properly and return it to you.
    
    note for Windows
    ~~~~~~~~
        
        multiprocessing in Python uses spawn meethod in MS Windows. This
        means that every time you have a new process the script that 
        contains the __main__ of your software will rerun.
        This means that in windows, you have to make sure the script 
        does not import anything heavy before __main__(). The main
        recommendation is that you basically have an actual code
        in a file named main.py and another simple file named
        after your application with no import or anything 
        in it except for the two following lines only:
        
        if __name__=='__main__':
            exec(main)

        as such the spawning in Windows will restart this file and when
        it reaches the if statement, it will let the process work.
        If you don't do this, you will see lots of time wasted around
        all the imports and if you are printing anything or if you have
        a GUI, you will see them repeat themselves for every process.

        Other OSs use fork.

    How to use write your function
    ~~~~~~~~~~~~
    
    You need a function that takes these inputs:
        1 - a single integer: the index of current processes
        optional: All inputs that you use by indexing
        2 - all inputs you send as a tuple and you intrepret them
            yourself...notice that for "READING" in multiprocessing the id
            of an object before sending it to processors is the same 
            inside each processor.
            This is NOT the case for "Writing to things", the output must
            be queued. Multiprocessing realizes if you write to something
            but does not keep track of those who you just read from.
        optional: All inputs that we have iterated over.
        3 - all iteratible inputs you send as a tuple and you intrepret them
            yourself..However, each memeber of the tuple must be indexable
            and the tuple that is sent to the function will be a tuple
            whose members are subsets of members of the original tuple 
            
    Example
    ~~~~~~~~~~~~
    
    A code snippet is brought here::
    
        from multiprocessor import multiprocessor
    
        def masked_median(idx, allInputs):
            data, mask, statistics_func = allInputs
            _data = data[idx]
            _mask = mask[idx]
            vector_to_analyze = _data[_mask==1]
            to_return = statistics_func(vector_to_analyze)
            return(np_array([to_return]))
        
        data_shape = (1000, 1000000)
        data = np.random.randn(*data_shape)
        mask = (2*np.random.rand(*data_shape)).astype('int')
        statistics_func = np.median
        
        allInputs = (data, mask, op_type)
        medians = multiprocessor(some_function, allInputs).start()
        
    """    
    def __init__(self, 
        targetFunction,
        indices = None,
        inputs = None,
        iteratable_inputs = None,
        max_cpu = None,
        batchSize = None,
        concatenateOutput = True,
        functionOutputSample = None, 
        functionOutputSample_timeout = 0,
        showProgress = False,
        test_mode = False):
        
        """
        input arguments
        ~~~~~~~~~~~~~~~
            targetFunction: Target function
            indices: should be the indices of indexable parts of your input
                example: if you have N processes, just give N,
                    but if you like to only processes certain indices 
                    send those in. We will be sending these indices
                    to your function, The first element of your function
                    will see each of these indices.
            inputs: all READ-ONLY inputs....notice: READ-ONLY 
            max_cpu: max number of allowed CPU
                default: None
            batchSize: how many data points are sent to each CPU at a time
                default: n_CPU/n_points/2
            concatenateOutput: If an output is np.ndarray and it can be
                concatenated along axis = 0, with this flag, we will
                put it as a whole ndarray in the output. Otherwise 
                the output will be a list.
            functionOutputSample: function output sample,  it is needed
                to know how the output of the function looks like to be able
                to concatenate them and produce a general output. For example
                if the output for a single trial is a list of numpy arrays and
                strings, give that to this input, 
                e.g.: list([np_zeros(100, 200), 'MyString'])
            functionOutputSample_timeout: 
                If you do not provide functionOutputSample
                we can call your function with index 0 to get it. 
                But it is possible and reasonable that the function takes 
                longer time than this input (default value is 0 seconds which
                means we don't run it.)
                Then we stop waiting for the function output. 
            showProgress: using textProgBar, it shows the progress of 
                multiprocessing of your task.
                default: False
        """
        self.test_mode = test_mode
        if iteratable_inputs is not None:
            iteratable_inputs = list(iteratable_inputs)
            
        if indices is None:
            assert iteratable_inputs is not None, \
                'multiprocessor:If indices are not provided,'\
                'iteratable_inputs must be a list of inputs '\
                'for me to iterate over.'
            try:
                indices = len(iteratable_inputs[0])
            except:
                pass
            if indices is None:
                try:
                    indices = iteratable_inputs[0].shape[0]
                except Exception as e:
                    raise Exception(
                        'You did not provide indices. I tried to get the length'
                        ' of iteratable_inputs using len or .shape[0] which'
                        ' did not work.'
                        ) from e
        else:
            try:
                indices = int(indices)
                if(showProgress):
                    print('Indices you gave will be np_arange(',indices,').')
                indices = np_arange(indices)
            except:
                pass
            if(not type(indices).__module__ == np___name__):
                try:
                    indices = np_array(indices).astype('int')
                    if(showProgress):
                        print('Input indices are turned into numpy array')
                except:
                    print('I can not interpret the input indices')
                    print('They are not numpy ints or cannot be turned into it.')
                    raise ValueError
        if((indices != indices.astype('int')).any()):
            print('Input indices are not integers?')
            raise ValueError
        
        indices = indices.astype('int64')
        self.n_pts = indices.shape[0]     
        if(showProgress):
            print('Input indices are a numpy int ndArray with ', 
                  self.n_pts, ' data points')
        if(inputs is not None):
            if(not type(inputs).__name__ == 'tuple'):
                inputs = (inputs, )
        if(iteratable_inputs is not None):
            if(not type(iteratable_inputs).__name__ == 'tuple'):
                iteratable_inputs = (iteratable_inputs, )

        # from here we try to determine the format of the output in case 
        # it is numpy ndarray
        FLAG_outputSampleIsAvailable = False
        if((functionOutputSample_timeout > 0) & (functionOutputSample is None)):
            if(showProgress):
                print('No output sample is provided, '
                      'I will call the function for a sample output.')
            try:
                _args = 0
                if(inputs is not None):
                    _args = (_args, ) + (inputs, )
                if(iteratable_inputs is not None):
                    if(not type(_args).__name__ == 'tuple'):
                        _args = (_args, )
                    it_input = ()
                    for iim in iteratable_inputs:
                        it_input = it_input + (iim[0], )
                    _args = _args + (it_input, )
                p = Process(target = targetFunction, args = _args)
                p.start()
                p.join(functionOutputSample_timeout)
                if p.is_alive():
                    p.kill()
                    p.join()
                    if(showProgress):
                        print('It takes too long to get a sample of the output of function')
                        print('Timeout threshold was ', functionOutputSample_timeout, 's.')
                        print('Consider giving me a sample of the output of the function,')
                        print('This is useful when its output is a numpy array')
                        print('I assume output is a list for now.')
                else:
                    if(showProgress):
                        print('I could call your given function with first data point')
                        print('Your function runs fast. Please consider that multiprocessing '+ \
                              'is most effective on functions that are slow and do a lot. It ' + \
                              'reduces the significance of the scheduling overhead.')
                    functionOutputSample = targetFunction(*_args)
                    if(functionOutputSample is None):
                        functionOutputSample = np_array([1])
                        print('The output is None.')
                    FLAG_outputSampleIsAvailable = True
            except Exception as e:
                print('I cannot run your function for index 0' \
                      'You need to make your function work as follows:' \
                      'The first input will be the index of data point.' \
                      'The second input can be all of your non-iteratable inputs' \
                      'The third input can be all of your iteratable inputs.')
                raise e

        if(not FLAG_outputSampleIsAvailable):
                functionOutputSample = []

        self.outputIsNumpy = False
        if(type(functionOutputSample).__module__ == np___name__):
            self.outputIsNumpy = True
            self.output_shape = functionOutputSample.shape
            self.output_dtype = functionOutputSample.dtype
            self.allResults = np_zeros(
                shape = ((self.n_pts,) + self.output_shape), 
                dtype = self.output_dtype)
            if(showProgress):
                print('output is a numpy ndArray')
                print('output_shape, ', self.output_shape)
                print('shape to prepare: ', ((self.n_pts,) + self.output_shape))
                print('allResults shape, ', self.allResults.shape)
        else:
            self.allResults = []
            self.Q_procID = np_array([], dtype='int')
            if(showProgress):
                print('output is a list with ') 
            self.output_types = []
        ###################################################################
        
        self.concatenateOutput = concatenateOutput
        self.indices = indices
        self.targetFunction = targetFunction
        self.inputs = inputs
        self.iteratable_inputs = iteratable_inputs
        self.showProgress = showProgress
        if(max_cpu is not None):
            self.max_cpu = max_cpu
        else:
            self.max_cpu = cpu_count() - 1  #Let's keep one for the queue handler
        self.default_batchSize = int(np_ceil(self.n_pts/self.max_cpu/2))
        if(batchSize is not None):
            if(self.default_batchSize >= batchSize):
                self.default_batchSize = batchSize
        if(showProgress):
            print('RGFLib multiprocessor initialized with:') 
            print('max_cpu: ', self.max_cpu)
            print('n_pts: ', self.n_pts)
            print('default_batchSize: ', self.default_batchSize)
            print('concatenateOutput: ', self.concatenateOutput)

    def _multiprocFunc(self, theQ, procID_range, _iteratable_inputs = None):
        if(self.outputIsNumpy):
            allResults = np_zeros(
                shape = ((procID_range.shape[0],) + self.output_shape), 
                dtype = self.output_dtype)
        else:
            allResults = []
        for idx, procCnt in enumerate(procID_range):
            funcIdx = self.indices[procCnt]
            try:
                _args = funcIdx
                if(self.inputs is not None):
                    _args = (_args, ) + (self.inputs, )
                if(_iteratable_inputs is not None):
                    if(not type(_args).__name__ == 'tuple'):
                        _args = (_args, )
                    it_input = ()
                    for iim in _iteratable_inputs:
                        it_input = it_input + (iim[idx], )
                    _args = _args + (it_input, )
                results = self.targetFunction(*_args)
                if(results is None):
                    results = np_array([1])
            except:
                print('Something in multiprocessing went wrong.')
                print('funcIdx-->.', funcIdx)
                exit()
            if(self.outputIsNumpy):
                allResults[idx] = results
            else:
                allResults.append(results)
        theQ.put(list([procID_range, allResults]))
        
    def start(self):
        aQ = Queue()
        numProc = self.n_pts
        procID = 0
        numProcessed = 0
        numBusyCores = 0
        firstProcess = True
        if(self.showProgress):
            print('Carrying on with the processing....')
        while(numProcessed<numProc):
            if (not aQ.empty()):
                aQElement = aQ.get()
                ret_procID_range = aQElement[0]
                _batchSize = ret_procID_range.shape[0]
                ret_result = aQElement[1]
                if(self.outputIsNumpy):
                    self.allResults[ret_procID_range] = ret_result
                else:
                    self.Q_procID = np_concatenate((self.Q_procID, 
                                                    ret_procID_range))
                    self.allResults += ret_result
                numProcessed += _batchSize
                numBusyCores -= 1
                if(self.showProgress):
                    if(firstProcess):
                        pBar = printprogress(numProc-1, title = 'starting ' \
                            + str(numProc) + ' processes with ' \
                            + str(self.max_cpu) + ' CPUs')
                        firstProcess = False
                    pBar(_batchSize)
            if((procID<numProc) & (numBusyCores < self.max_cpu)):
                batchSize = np_minimum(self.default_batchSize, numProc - procID)
                procID_arange = np_arange(procID, procID + batchSize, dtype = 'int')
                _args = (aQ, procID_arange, )

                if(self.iteratable_inputs is not None):
                    it_input = ()
                    for iim in self.iteratable_inputs:
                        it_input = it_input + (iim[procID_arange], )
                    _args = _args + (it_input, )
                if(self.test_mode):
                    self._multiprocFunc(*_args)
                else:
                    Process(target = self._multiprocFunc, 
                                      args = _args).start()
                procID += batchSize
                numBusyCores += 1
        
        if(self.outputIsNumpy):
            return (self.allResults)
        else:
            n_individualOutputs = len(self.allResults[0])
            sortArgs = np.argsort(self.Q_procID)
            ret_list = [self.allResults[i] for i in sortArgs]
            endResultList = []
            for memberCnt in range(n_individualOutputs):
                FLAG_output_is_numpy = False
                if(self.concatenateOutput):
                    firstInstance = ret_list[0][memberCnt]
                    if(type(firstInstance).__module__ == np___name__):
                        if(isinstance(firstInstance, np.ndarray)):
                            n_F = 0
                            for ptCnt in range(0, self.n_pts):
                                n_F += ret_list[ptCnt][memberCnt].shape[0]
                            outShape = ret_list[ptCnt][memberCnt].shape[1:]
                            _currentList = np_zeros(
                                shape = ( (n_F,) + outShape ), 
                                dtype = ret_list[0][memberCnt].dtype)
                            n_F = 0
                            for ptCnt in range(0, self.n_pts):
                                ndarr = ret_list[ptCnt][memberCnt]
                                _n_F = ndarr.shape[0]
                                _currentList[n_F: n_F + _n_F] = ndarr
                                n_F += _n_F
                            FLAG_output_is_numpy = True
                        else:
                            print('Output member', memberCnt, 'could not be',
                                'concatenated along axis 0. exported as list.',
                                '\nThis usually happens when you use ',
                                'np.mean(), np.std() or np.median(), Still numpy but not arrays',
                                ' Make sure you present it as np_array([np.mean()]).')
                if(not FLAG_output_is_numpy):
                    _currentList = []
                    for ptCnt in range(self.n_pts):
                        _currentList.append(ret_list[ptCnt][memberCnt])
                endResultList.append(_currentList)
            return (endResultList)