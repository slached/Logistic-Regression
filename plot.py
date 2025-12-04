import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np

from matplotlib import rcParams
rcParams.update({'font.size': 18, 'text.usetex': True})

logreg_GD_weights, logreg_GD_objective = pkl.load(open('./results/logreg_GD.pkl', 'rb'))

def plot_logreg():
    '''
    logistic regression: weights
    '''

    logreg_dimension = 785
    plt.figure()
    plt.plot(range(len(logreg_GD_weights)), np.array(logreg_GD_weights) / np.sqrt(logreg_dimension), label = 'GD')

    plt.legend()

    plt.xlabel('Iterations')
    plt.ylabel(r'$\frac{1}{\sqrt{d}}\|x^{(k)}-x^{\star}\|_2$')
    plt.title('Logistic Regression')

    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('./results/logreg_weights.png', dpi=1200)
    plt.show()
    plt.pause(5)


    '''
    logistic regression: objective
    '''

    plt.figure()
    plt.plot(range(len(logreg_GD_objective)), logreg_GD_objective, label = 'GD')

    plt.legend()

    plt.xlabel('Iterations')
    plt.ylabel(r'$f(x^{(k)}) - p^{\star}$')
    plt.title('Logistic Regression')

    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('./results/logreg_objective.png', dpi=1200)
    plt.show()

if __name__ == '__main__':

    plot_logreg()
    
    

    