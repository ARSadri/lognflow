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
from numpy import unique      as np_unique

from multiprocessing import Process, Queue, cpu_count, Event

from .printprogress import printprogress
from .utils import is_builtin_collection

def _to_collection(returned_obj):
    if not is_builtin_collection(returned_obj):
        return [returned_obj]
    return returned_obj

def _multiprocessor_function_test_mode(
        iterables_batch, targetFunction, \
        shareables, theQ, procID_range, error_event):
    outputs = []
    for idx, procCnt in enumerate(procID_range):
        if len(iterables_batch) == 1:
            iterables_sliced = iterables_batch[0][idx]
        else:
            iterables_sliced = ()
            for iim in iterables_batch:
                iterables_sliced = iterables_sliced + (iim[idx], )
        if shareables is None:
            results = targetFunction(iterables_sliced)
        else:
            results = targetFunction(iterables_sliced, shareables)
        outputs.append(_to_collection(results))
    theQ.put([procID_range, outputs, False])

def _multiprocessor_function(iterables_batch, targetFunction, \
        shareables, theQ, procID_range, error_event):
    outputs = []
    for idx, procCnt in enumerate(procID_range):
        try:
            assert not error_event.is_set()
            if len(iterables_batch) == 1:
                iterables_sliced = iterables_batch[0][idx]
            else:
                iterables_sliced = ()
                for iim in iterables_batch:
                    iterables_sliced = iterables_sliced + (iim[idx], )
            if shareables is None:
                results = targetFunction(iterables_sliced)
            else:
                results = targetFunction(iterables_sliced, shareables)
            outputs.append(_to_collection(results))
        except Exception as e:
            if not error_event.is_set():
                error_event.set()
            theQ.put([np_array([procCnt]), None, True])
            return
    theQ.put([procID_range, outputs, False])

def _prepare_outpus(outputs, Q_procID, concatenate_outputs, outputs_is_given):
    if(outputs_is_given):
        return outputs
    else:
        sortArgs = np_argsort(Q_procID)
        ret_Q_procID = [Q_procID[i] for i in sortArgs]
        ret_list = [outputs[i] for i in sortArgs]

        return_as_is = False
        ret_entries_lens = []
        for ret_entry in ret_list:
            try:
                _len = len(ret_entry)
            except:
                try:
                    _len = ret_entry.size
                except:
                    return_as_is = True
                else:
                    ret_entries_lens.append(_len)
            else:
                ret_entries_lens.append(_len)
        ret_entries_lens_unique = np_unique(ret_entries_lens)
        if len(ret_entries_lens_unique) != 1:
            return_as_is = True
        
        if return_as_is | (not concatenate_outputs):
            return ret_list
        else:
            n_entries = ret_entries_lens_unique[0]
            outputs = []
            for element_cnt in range(n_entries):
                element_all = []
                is_not_nparray = False
                is_numpy = False
                shapes_are_not_the_same = False
                shapes_are_the_same = False
                for entry in ret_list:
                    instance = entry[element_cnt]
                    try:
                        instance_size = instance.size
                    except:
                        is_not_nparray = True
                    else:
                        if instance_size == 0:
                            is_not_nparray = True
                        else:
                            if not is_numpy:
                                numpy_shape = instance.shape
                            else:
                                if numpy_shape == instance.shape:
                                    shapes_are_the_same = True
                                else:
                                    shapes_are_not_the_same = True
                            is_numpy = True
                    element_all.append(instance)
                if ((not is_not_nparray) & 
                    is_numpy & 
                    (not shapes_are_not_the_same) &
                    shapes_are_the_same):
                    element_all = np_array(element_all)
                outputs.append(element_all)  
            if n_entries == 1:
                return outputs[0]
            else:
                return outputs

def _reraise_any_error(
        any_error, targetFunction, error_ret_procID, iterables, 
        shareables, aQ, error_event, logger):            
    logger('-'*79)
    logger('An exception occured during submitting jobs.')
    logger('Here we try to reproduce it but will raise '
          'ChildProcessError regardless.')
    logger(f'We will call {targetFunction} ')
    logger('with the following index to slice the inputs:'
          f' {error_ret_procID[0]}')
    logger('to avoid seeing this message, pass the argument called '\
           'logger, it is print by default.')
    logger('-'*79)
    iterables_batch = ()
    for iim in iterables:
        iterables_batch = \
            iterables_batch + (iim[error_ret_procID], )
    _args = (iterables_batch, ) + (
        targetFunction, shareables, aQ, error_ret_procID, error_event)
    _multiprocessor_function_test_mode(*_args)
    raise ChildProcessError
    
def multiprocessor(
    targetFunction,
    iterables,
    shareables          = None,
    outputs             = None,
    max_cpu             = None,
    batchSize           = None,
    concatenate_outputs = True,
    verbose             = False,
    test_mode           = False,
    logger              = print):
    """ multiprocessor makes the use of multiprocessing in Python easy and fast
    
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
        iterables_sliced:
            When providing iterables, we will send a single element of every
            member of it to the function. If it is a numpy array, we will send
            iterables[i] to your function. if it is a tuple of a few arrays,
            we send a tuple of a few slices: (arr[i], brr[i], ...)
        shareables: All inputs that we are just passed to your function.
    
    Example
    ~~~~~~~~~~~~
    
    A code snippet is brought here::
    
        from lognflow import multiprocessor
    
        def masked_cross_correlation(iterables_sliced, shareables):
            vec1, vec2 = iterables_sliced
            mask, statistics_func = shareables
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
        
        iterables = (data1, data2)
        shareables = (mask, op_type)
        ccorr = multiprocessor(some_function, iterables, shareables)
        print(f'ccorr: {ccorr}')
        
    input arguments
    ~~~~~~~~~~~~~~~
        targetFunction: Target function
        iterables: all iterabel inputs, We will pass them by indexing
            them. if indices are not provideed, the len(iterables[0])
            will be N.
        shareables: all READ-ONLY inputs.... Notice: READ-ONLY 
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
    if shareables is not None:
        if not isinstance(shareables, tuple):
            shareables = (shareables, )
    
    try:
        n_pts = int(iterables)
        assert n_pts == iterables, \
            'if iterables is a single number, please provide an integer.'
        iterables = [np_arange(n_pts, dtype='int')]
    except:
        try:
            n_pts = iterables.shape[0]
            iterables = [iterables]
        except:
            try:
                n_pts = len(iterables[0])
            except:
                try:
                    n_pts = iterables[0].shape[0]
                except Exception as e:
                    raise Exception(
                        'You did not provide iterables properly.'
                        ' It should be either a list or tuple where all members'
                        ' have the same length (first dimensions) or it can be'
                        ' a numpy array to iterate over, or it can be an'
                        ' integer.'
                        ) from e
    indices = np_arange(n_pts, dtype='int')
    if(verbose):
        logger(f'inputs to iterate over are {n_pts}.')

    if(max_cpu is None):
        max_cpu = cpu_count()
    default_batchSize = int(np_ceil(n_pts/max_cpu/2))
    if(batchSize is not None):
        if(default_batchSize >= batchSize):
            default_batchSize = batchSize
    if(verbose):
        logger('lognflow multiprocessor initialized with:') 
        logger('max_cpu: ', max_cpu)
        logger('n_pts: ', n_pts)
        logger('default_batchSize: ', default_batchSize)
        logger('concatenate_outputs: ', concatenate_outputs)

    aQ = Queue()

    if(outputs is None):
        outputs_is_given = False
        outputs = []
    else:
        outputs_is_given = True
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
                    for ret_procID_range_element in ret_procID_range:
                        Q_procID.append(ret_procID_range_element)
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
            procID_range = np_arange(procID, procID + batchSize, dtype = 'int')

            iterables_batch = ()
            for iim in iterables:
                iterables_batch = \
                    iterables_batch + (iim[procID_range], )
            _args = (iterables_batch, ) + (
                targetFunction, shareables, aQ, procID_range, error_event)
            
            if(test_mode):
                _multiprocessor_function_test_mode(*_args)
            else:
                Process(target = _multiprocessor_function, args = _args).start()
            procID += len(procID_range)
            numBusyCores += 1
    
    if(any_error):        
        _reraise_any_error(
            any_error, targetFunction, error_ret_procID, iterables, 
            shareables, aQ, error_event, logger)
    
    return _prepare_outpus(
        outputs, Q_procID, concatenate_outputs, outputs_is_given)

def multiprocessor_gen(
    targetFunction,
    iterables,
    shareables          = None,
    outputs             = None,
    max_cpu             = None,
    batchSize           = None,
    concatenate_outputs = True,
    verbose             = False,
    test_mode           = False,
    logger              = print):
    """ multiprocessor_gen makes the use of multiprocessing in Python easy
    
        It is exactly the same as multiprocessor, however, it yields the
        2-tuple of (results, ID) when a result arrives. You have to use it
        like a normal generator in a for loop. Look at the above multiprocessor
        for documentation on parameters and look at the tests for 
        multiprocessor_gen for an example how to use a generator in Python.
    """
    if shareables is not None:
        if not isinstance(shareables, tuple):
            shareables = (shareables, )
    
    try:
        n_pts = int(iterables)
        assert n_pts == iterables, \
            'if iterables is a single number, please provide an integer.'
        iterables = [np_arange(n_pts, dtype='int')]
    except:
        try:
            n_pts = iterables.shape[0]
            iterables = [iterables]
        except:
            try:
                n_pts = len(iterables[0])
            except:
                try:
                    n_pts = iterables[0].shape[0]
                except Exception as e:
                    raise Exception(
                        'You did not provide iterables properly.'
                        ' It should be either a list or tuple where all members'
                        ' have the same length (first dimensions) or it can be'
                        ' a numpy array to iterate over, or it can be an'
                        ' integer.'
                        ) from e
    indices = np_arange(n_pts, dtype='int')
    if(verbose):
        logger(f'inputs to iterate over are {n_pts}.')

    if(max_cpu is None):
        max_cpu = cpu_count()
    default_batchSize = int(np_ceil(n_pts/max_cpu/2))
    if(batchSize is not None):
        if(default_batchSize >= batchSize):
            default_batchSize = batchSize
    if(verbose):
        logger('lognflow multiprocessor initialized with:') 
        logger('max_cpu: ', max_cpu)
        logger('n_pts: ', n_pts)
        logger('default_batchSize: ', default_batchSize)
        logger('concatenate_outputs: ', concatenate_outputs)

    aQ = Queue()

    if(outputs is None):
        outputs_is_given = False
        outputs = []
    else:
        outputs_is_given = True
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
                    for ret_procID_range_element in ret_procID_range:
                        Q_procID.append(ret_procID_range_element)
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
            
            if(not any_error):
                _outputs = _prepare_outpus(
                    outputs, Q_procID, concatenate_outputs, outputs_is_given)
                yield _outputs, Q_procID
                
        if((procID<n_pts) & (numBusyCores < max_cpu) & (not any_error)):
            batchSize = np_minimum(default_batchSize, n_pts - procID)
            procID_range = np_arange(procID, procID + batchSize, dtype = 'int')

            iterables_batch = ()
            for iim in iterables:
                iterables_batch = \
                    iterables_batch + (iim[procID_range], )
            _args = (iterables_batch, ) + (
                targetFunction, shareables, aQ, procID_range, error_event)
            
            if(test_mode):
                _multiprocessor_function_test_mode(*_args)
            else:
                Process(target = _multiprocessor_function, args = _args).start()
            procID += len(procID_range)
            numBusyCores += 1
    
    if(any_error):        
        _reraise_any_error(
            any_error, targetFunction, error_ret_procID, iterables, 
            shareables, aQ, error_event, logger)
