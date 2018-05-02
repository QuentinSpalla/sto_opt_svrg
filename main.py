import constants
from get_data import AllData
from svrm import SVRG
import matplotlib.pyplot as plt

all_data = AllData(constants.NBR_SAMPLES, constants.NBR_FEATURES)
my_svrg = SVRG(all_data.A, all_data.b, constants.NBR_EPOCH, constants.NBR_UPDATES, constants.INITIAL_LEARNING_RATE)
my_svrg.train()

plt.plot(my_svrg.grad_history, label='SVRG')
plt.grid()
plt.yscale('log')
plt.show()
