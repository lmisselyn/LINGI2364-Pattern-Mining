import numpy as np
import matplotlib.pyplot as plt

def plot(acc):
    x = np.arange(1, 11)
    plt.bar(x, acc, width=0.5)
    plt.xlabel('number of k patterns returned')
    plt.ylabel('accuracy')
    plt.grid(axis='y')
    plt.xticks(x)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.title("Alternative with Wracc scoring (Reuter)")
    plt.show()

if __name__ == '__main__':
    spade_reuter = [0.651, 0.654, 0.654, 0.724, 0.724, 0.723, 0.856, 0.856, 0.856, 0.861]
    spade_prot_supp = [0.673, 0.813, 0.813, 0.813, 0.791, 0.787, 0.806, 0.894, 0.873, 0.869]
    spade_wracc_reut = [0.828, 0.828, 0.828, 0.828, 0.899, 0.899, 0.901, 0.901, 0.901, 0.899]
    spade_wracc_prot = [0.858, 0.858, 0.858, 0.876, 0.888, 0.882, 0.894, 0.898, 0.895, 0.895]

    alt_reut_supp = [0.659, 0.659, 0.824, 0.826, 0.877, 0.886, 0.88, 0.883, 0.89, 0.888]
    alt_prot_supp = [0.688, 0.86, 0.86, 0.86, 0.86, 0.86, 0.86, 0.86, 0.86, 0.86]
    alt_reut_wracc = [0.824, 0.895, 0.895, 0.905, 0.911, 0.923, 0.931, 0.928, 0.931, 0.937]
    alt_prot_wracc = [0.860, 0.860, 0.860, 0.860, 0.860, 0.860, 0.860, 0.860, 0.860, 0.860]
    plot(alt_reut_wracc)
