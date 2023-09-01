from lognflow import multiprocessor
import numpy as np

def multiprocessor_targetFunc(idx, inputs):
    data, mask, op_type = inputs
    _data = data[idx]
    if(op_type=='median'):
        to_return1 = np.median(_data[mask[idx]==1])
        to_return1 = np.array([to_return1])
    to_return2 = np.ones((int(10*np.random.rand(1)), 2, 2))
    
    return(to_return1, 'median', to_return2)
    
def test_multiprocessor():
    N = 10000
    D = 10000
    data = (10+100*np.random.randn(N,D)).astype('int')
    mask = (2*np.random.rand(N,D)).astype('int')
    op_type = 'median'

    inputs  = (data, mask, op_type)

    from multiprocessor import multiprocessor
    stats = multiprocessor(
        multiprocessor_targetFunc, N, inputs,
        showProgress = True).start()

    results = []
    for cnt in range(N):
        results.append(multiprocessor_targetFunc(cnt, inputs))

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

if __name__ == '__main__':
    print('lets test', flush=True)
    test_multiprocessor()