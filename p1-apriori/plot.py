import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

barWidth = 0.4
min_freq = ['0.8', '0.85', '0.9', '0.95', '1']
x_ap = np.arange(len(min_freq))
x_alt = [x + barWidth for x in x_ap]
alt_time = [0.716, 0.850, 0.662, 0.680, 0.778]
ap_time = [0.715, 0.667, 0.923, 0.863, 0.838]

fig, ax = plt.subplots()
ax.bar(x_ap, ap_time, width=0.4, label='Apriori')
ax.bar(x_alt, alt_time, width=0.4, label="ECLAT")
plt.xticks([r + 0.2 for r in range(len(min_freq))], min_freq)
ax.set_xlabel('Minimum frequency')
ax.set_ylabel('Execution time (sec)')
plt.legend()
ax.set_title('Execution time for each function - retail.dat')
plt.show()

if __name__ == '__main__':
    print('ok')