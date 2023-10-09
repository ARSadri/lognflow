from lognflow import multiprocessor
import numpy as np
import inspect

def multiprocessor_targetFunc(inputs_to_iter_sliced, inputs_to_share):
    idx = inputs_to_iter_sliced[0]
    data, mask, op_type = inputs_to_share
    _data = data[idx]
    if(op_type=='median'):
        to_return1 = np.median(_data[mask[idx]==1])
        to_return1 = np.array([to_return1])
    to_return2 = np.ones((int(10*np.random.rand(1)), 2, 2))
    
    return(to_return1, 'median', to_return2)
    
def test_multiprocessor():
    print('-'*80, '\n', inspect.stack()[0][3], '\n', '-'*80)

    N = 10000
    D = 1000
    data = (10+100*np.random.randn(N,D)).astype('int')
    mask = (2*np.random.rand(N,D)).astype('int')
    op_type = 'median'

    inputs_to_share  = (data, mask, op_type)
    inputs_to_iter = N
    
    stats = multiprocessor(
        multiprocessor_targetFunc, inputs_to_iter, inputs_to_share,
        verbose = True)

    results = []
    for cnt in range(N):
        results.append(multiprocessor_targetFunc((cnt, ), inputs_to_share))

    medians, otherOutput, _ids = stats
    print('type(medians)', type(medians))
    print('medians.shape', medians.shape)
    print('type(otherOutput)', type(otherOutput))
    print('len(otherOutput)', len(otherOutput))
    print('otherOutput[1] ', otherOutput[1])
    print('otherOutput[1][0] ', otherOutput[1][0])
    print('type(_ids) ', type(_ids))
    print('len(_ids) ', len(_ids))
    print('type(_ids[0]) ', type(_ids[0]))
    print('_ids.shape ', _ids.shape)
    
    direct_medians = np.zeros(N)
    for cnt in range(N):
        direct_medians[cnt] = np.median(data[cnt, mask[cnt]==1])
    
    print(np.array([ medians, direct_medians] ).T)
    print('difference of results: ', (direct_medians - medians).sum())

def masked_cross_correlation(inputs_to_iter_sliced, inputs_to_share):
    vec1, vec2 = inputs_to_iter_sliced
    mask, statistics_func = inputs_to_share
    vec1 = vec1[mask==1]
    vec2 = vec2[mask==1]
    
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

def test_multiprocessor_ccorr():
    print('-'*80, '\n', inspect.stack()[0][3], '\n', '-'*80)
    data_shape = (1000, 2000)
    data1 = np.random.randn(*data_shape)
    data2 = 2 + 5 * np.random.randn(*data_shape)
    mask = (2*np.random.rand(data_shape[1])).astype('int')
    statistics_func = np.median
    
    inputs_to_iter = (data1, data2)
    inputs_to_share = (mask, statistics_func)
    ccorr = multiprocessor(
        masked_cross_correlation, inputs_to_iter, inputs_to_share,
        test_mode = False)

def error_multiprocessor_targetFunc(inputs_to_iter_sliced, inputs_to_share):
    idx = inputs_to_iter_sliced[0]
    data, mask, op_type = inputs_to_share
    _data = data[idx]
    if(op_type=='median'):
        to_return1 = np.median(_data[mask[idx]==1])
        to_return1 = np.array([to_return1])
    to_return2 = np.ones((int(10*np.random.rand(1)), 2, 2))
    
    if idx == 3000:
        dsfee
    
    return(to_return1, 'median', to_return2)    

def test_error_handling_in_multiprocessor():
    print('-'*80, '\n', inspect.stack()[0][3], '\n', '-'*80)
    
    N = 10000
    D = 1000
    data = (10+100*np.random.randn(N,D)).astype('int')
    mask = (2*np.random.rand(N,D)).astype('int')
    op_type = 'median'

    inputs_to_share  = (data, mask, op_type)
    inputs_to_iter = N
    
    stats = multiprocessor(
        error_multiprocessor_targetFunc, inputs_to_iter, inputs_to_share,
        verbose = True)

if __name__ == '__main__':
    print('lets test', flush=True)
    try:
        test_error_handling_in_multiprocessor()
    except Exception as e:
        print('Error handled!')
        print('Error was:')
        print(e)
    test_multiprocessor()
    test_multiprocessor_ccorr()
