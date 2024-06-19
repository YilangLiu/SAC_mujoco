import numpy as np
import matplotlib.pyplot as plt
if __name__ == "__main__":
    z = np.loadtxt('z.txt',delimiter=',')
    v = np.loadtxt('v.txt',delimiter=',')
    lamb = np.loadtxt('lam.txt', delimiter=',')

    f, (ax1)= plt.subplots(1,1)
    ax1.plot(z)
    ax1.stem(v, 'k')
    ax1.legend(['position','angle','velocity','angular velocity'])
    #
    # ax2.plot(np.arange(11), lamb[5,:])

    plt.show()