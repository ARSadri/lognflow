from lognflow.plt_utils import plt, np, plt_imhist
from lognflow import printv

if __name__ == '__main__':
    if code_block_id == 1:
        img = np.random.randn(100, 100)
        img[30:50, 30:50] += 10
        plt_imhist(img); plt.show()
    
    if code_block_id == 2:    
        img[img > 50] *= img[img > 50]
        plt_imhist(img); plt.show()
    
    if code_block_id == 3:    
        img[img < 50] = np.exp(-img[img < 50])
        plt_imhist(img); plt.show() 
    
    if code_block_id == 4:    
        img += np.random.randn(100, 100)
        plt_imhist(img); plt.show() 
        
    if code_block_id == 5:    
        img = np.exp(-img)
        plt_imhist(img); plt.show() 
        
    printv(img)