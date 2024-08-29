from lognflow.plt_utils import plt, np, plt_imhist
import numpy as np

if pyrunner_cell_no == 1:
    vec = [1, 2, 3]

if pyrunner_cell_no == 2:    
    vec = [i ** 2 for i in vec]
    print("Squared vec:", vec)

if pyrunner_cell_no == 3:    
    vec = [np.exp(-i ** 2) for i in vec]
    print("Squared vec:", vec)
    plt_imhist(np.random.randn(100, 100))
    plt.show()

if pyrunner_cell_no == 4:    
    vec = [np.exp(i) for i in vec]
    print("Squared vec:", vec)

if pyrunner_cell_no == 5:    
    vec = [np.log(i) for i in vec]
    print("Squared vec:", vec)
    
print(f"Current state of vec: {vec}")