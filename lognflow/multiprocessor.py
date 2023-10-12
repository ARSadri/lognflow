#from-import is necessary to speed up spawning in Windows as much as possible
from numpy import __name__    as np___name__
from numpy import array       as np_array
from numpy import ndarray     as np_ndarray
from numpy import ceil        as np_ceil
from numpy import arange      as np_arange
from numpy import zeros       as np_zeros
from numpy import minimum     as np_minimum
from numpy import concatenate as np_concatenate
from numpy import argsort     as np_argsort

from multiprocessing import Process, Queue, cpu_count, Event

from .printprogress import printprogress

def _multiprocessor_function_test_mode(
        inputs_to_iter_batch, targetFunction, \
        inputs_to_share, theQ, procID_range, error_event):
    outputs = []
    for idx, procCnt in enumerate(procID_range):
        inputs_to_iter_sliced = ()
        for iim in inputs_to_iter_batch:
            inputs_to_iter_sliced = inputs_to_iter_sliced + (iim[idx], )
        if inputs_to_share is None:
            results = targetFunction(inputs_to_iter_sliced)
        else:
            results = targetFunction(inputs_to_iter_sliced, inputs_to_share)
        outputs.append(results)
    theQ.put([procID_range, outputs, False])

def _multiprocessor_function(inputs_to_iter_batch, targetFunction, \
        inputs_to_share, theQ, procID_range, error_event):
    outputs = []
    for idx, procCnt in enumerate(procID_range):
        inputs_to_iter_sliced = ()
        for iim in inputs_to_iter_batch:
            inputs_to_iter_sliced = inputs_to_iter_sliced + (iim[idx], )
        try:
            assert not error_event.is_set()
            if inputs_to_share is None:
                results = targetFunction(inputs_to_iter_sliced)
            else:
                results = targetFunction(inputs_to_iter_sliced, inputs_to_share)
            outputs.append(results)
        except Exception as e:
            if not error_event.is_set():
                error_event.set()
            theQ.put([np_array([procCnt]), None, True])
            return
    theQ.put([procID_range, outputs, False])

def multiprocessor(
    targetFunction,
    inputs_to_iter,
    inputs_to_share     = None,
    outputs             = None,
    max_cpu             = None,
    batchSize           = None,
    concatenate_outputs = True,
    verbose             = False,
    test_mode           = False,
    logger              = print):
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
    You need a function that takes two inputs:
        inputs_to_iter_sliced:
            When providing inputs_to_iter, we will send a single element of every
            member of it to the function. If it is a numpy array, we will send
            inputs_to_iter[i] to your function. if it is a tuple of a few arrays,
            we send a tuple of a few slices: (arr[i], brr[i], ...)
        inputs_to_share: All inputs that we are just passed to your function.
    
    Example
    ~~~~~~~~~~~~
    
    A code snippet is brought here::
    
        from lognflow import multiprocessor
    
        def masked_cross_correlation(inputs_to_iter_sliced, inputs_to_share):
            vec1, vec2 = inputs_to_iter_sliced
            mask, statistics_func = inputs_to_share
            vec1 = vec1[_mask==1]
            vec2 = vec2[_mask==1]
            
            vec1 -= vec1.mean()
            vec1_std = vec1.std()
            if vec1_std > 0:
                vec1 /= vec1_std
            vec2 -= vec2.mean()
            vec2_std = vec2.std()
            if vec2_std > 0:
                vec2 /= vec2_std

            correlation = vec1 * vec2
            to_return = statistics_func(correlation)
            return(to_return)
        
        data_shape = (1000, 1000000)
        data1 = np.random.randn(*data_shape)
        data2 = 2 + 5 * np.random.randn(*data_shape)
        mask = (2*np.random.rand(*data_shape)).astype('int')
        statistics_func = np.median
        
        inputs_to_iter = (data1, data2)
        inputs_to_share = (mask, op_type)
        ccorr = multiprocessor(some_function, inputs_to_iter, inputs_to_share)
        print(f'ccorr: {ccorr}')
        
    input arguments
    ~~~~~~~~~~~~~~~
        targetFunction: Target function
        inputs_to_iter: all iterabel inputs, We will pass them by indexing
            them. if indices are not provideed, the len(inputs_to_iter[0])
            will be N.
        inputs_to_share: all READ-ONLY inputs.... Notice: READ-ONLY 
        outputs: an indexable memory where we can just dump the output of 
            function in relevant indices.  For example a numpy
        max_cpu: max number of allowed CPU
            default: None
        batchSize: how many data points are sent to each CPU at a time
            default: n_CPU/n_points/2
        concatenate_outputs: If an output is np.ndarray and it can be
            concatenated along axis = 0, with this flag, we will
            put it as a whole ndarray in the output. Otherwise 
            the output will be a list.
        verbose: using textProgBar, it shows the progress of 
            multiprocessing of your task.
            default: False
    """
    if inputs_to_share is not None:
        if not isinstance(inputs_to_share, tuple):
            inputs_to_share = (inputs_to_share, )
    
    try:
        n_pts = int(inputs_to_iter)
        assert n_pts == inputs_to_iter, \
            'if inputs_to_iter is a single number, please provide an integer.'
        inputs_to_iter = [np_arange(n_pts, dtype='int')]
    except:
        try:
            n_pts = inputs_to_iter.shape[0]
            inputs_to_iter = [inputs_to_iter]
        except:
            try:
                n_pts = len(inputs_to_iter[0])
            except:
                try:
                    n_pts = inputs_to_iter[0].shape[0]
                except Exception as e:
                    raise Exception(
                        'You did not provide inputs_to_iter properly.'
                        ' It should be either a list or tuple where all members'
                        ' have the same length (first dimensions) or it can be'
                        ' a numpy array to iterate over, or it can be an'
                        ' integer.'
                        ) from e
    indices = np_arange(n_pts, dtype='int')
    if(verbose):
        logger(f'inputs to iterate over are {n_pts}.')

    if(max_cpu is None):
        max_cpu = cpu_count() - 1  #Let's keep one for the OS
    default_batchSize = int(np_ceil(n_pts/max_cpu/2))
    if(batchSize is not None):
        if(default_batchSize >= batchSize):
            default_batchSize = batchSize
    if(verbose):
        logger('RGFLib multiprocessor initialized with:') 
        logger('max_cpu: ', max_cpu)
        logger('n_pts: ', n_pts)
        logger('default_batchSize: ', default_batchSize)
        logger('concatenate_outputs: ', concatenate_outputs)

    aQ = Queue()

    outputs_is_given = True
    if(outputs is None):
        outputs_is_given = False
        outputs = []
        Q_procID = []
    
    procID = 0
    numProcessed = 0
    numBusyCores = 0
    if(verbose):
        pBar = printprogress(n_pts, title = 
            f'Processing {n_pts} data points with {max_cpu} CPUs')
    any_error = False
    
    error_event = Event()
    
    while(numProcessed<n_pts):
        if (not aQ.empty()):
            aQElement = aQ.get()
            ret_procID_range = aQElement[0]
            ret_result = aQElement[1]
            if ((not any_error) & aQElement[2]):
                any_error = True
                error_ret_procID = ret_procID_range.copy()
                try:
                    pBar._end()
                except:
                    pass
                logger('lognflow, multiprocessor:')
                logger('An exception has been raised. Joining all processes...')
            if (not any_error):
                if(outputs_is_given):
                    outputs[ret_procID_range] = ret_result
                else:
                    for ret_procID, result in zip(ret_procID_range, ret_result):
                        Q_procID.append(ret_procID)
                        outputs.append(result)
            else:
                logger(f'Number of busy cores: {numBusyCores}')

            _batchSize = ret_procID_range.shape[0]
            numProcessed += _batchSize
            numBusyCores -= 1
            if(verbose & (not any_error)):
                pBar(_batchSize)
            if(any_error & (numBusyCores == 0)):
                logger(f'All cores are free')
                break
                
        if((procID<n_pts) & (numBusyCores < max_cpu) & (not any_error)):
            batchSize = np_minimum(default_batchSize, n_pts - procID)
            procID_arange = np_arange(procID, procID + batchSize, dtype = 'int')

            inputs_to_iter_batch = ()
            for iim in inputs_to_iter:
                inputs_to_iter_batch = \
                    inputs_to_iter_batch + (iim[procID_arange], )
            _args = (inputs_to_iter_batch, ) + (
                targetFunction, inputs_to_share, aQ, procID_arange, error_event)
            
            if(test_mode):
                _multiprocessor_function_test_mode(*_args)
            else:
                Process(target = _multiprocessor_function, args = _args).start()
            procID += batchSize
            numBusyCores += 1
    
    if(any_error):
        logger('-'*79)
        logger('An exception occured during submitting jobs.')
        logger('Here we try to reproduce it but will raise '
              'ChildProcessError regardless.')
        logger(f'We will call {targetFunction} ')
        logger('with the following index to slice the inputs:'
              f' {error_ret_procID[0]}')
        logger('to avoid seeing this message, pass the argument called '\
               'legger, it is print by default.')
        logger('-'*79)
        inputs_to_iter_batch = ()
        for iim in inputs_to_iter:
            inputs_to_iter_batch = \
                inputs_to_iter_batch + (iim[error_ret_procID], )
        _args = (inputs_to_iter_batch, ) + (
            targetFunction, inputs_to_share, aQ, error_ret_procID, error_event)
        _multiprocessor_function_test_mode(*_args)
        raise ChildProcessError
    
    if(outputs_is_given):
        return outputs
    else:
        sortArgs = np_argsort(Q_procID)
        ret_list = [outputs[i] for i in sortArgs]
        firstInstance = ret_list[0]
        if(  (not isinstance(firstInstance, list))
           | (not isinstance(firstInstance, tuple))
           | (not isinstance(firstInstance, dict))):
            if(type(firstInstance).__module__ == np___name__):
                outputs = np_array(ret_list)
                return outputs
        
        n_individualOutputs = len(ret_list[0])
        outputs = []
        for memberCnt in range(n_individualOutputs):
            FLAG_output_is_numpy = False
            if(concatenate_outputs):
                firstInstance = ret_list[0][memberCnt]
                if(type(firstInstance).__module__ == np___name__):
                    try:
                        _ = firstInstance.shape
                    except:
                        firstInstance = np.array([firstInstance])
                    n_F = 0
                    for ptCnt in range(0, n_pts):
                        n_F += ret_list[ptCnt][memberCnt].shape[0]
                    outShape = ret_list[ptCnt][memberCnt].shape[1:]
                    _currentList = np_zeros(
                        shape = ( (n_F,) + outShape ), 
                        dtype = ret_list[0][memberCnt].dtype)
                    n_F = 0
                    for ptCnt in range(0, n_pts):
                        ndarr = ret_list[ptCnt][memberCnt]
                        _n_F = ndarr.shape[0]
                        _currentList[n_F: n_F + _n_F] = ndarr
                        n_F += _n_F
                    FLAG_output_is_numpy = True
            if(not FLAG_output_is_numpy):
                _currentList = []
                for ptCnt in range(n_pts):
                    _currentList.append(ret_list[ptCnt][memberCnt])
            outputs.append(_currentList)
        return (outputs)