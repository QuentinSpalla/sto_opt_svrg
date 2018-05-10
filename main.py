import constants
from get_data import AllData
from svrm import SVRG, CM, SVRG2, AM
import matplotlib.pyplot as plt

"""
all_data = AllData(constants.NBR_SAMPLES, constants.NBR_FEATURES)
"""

"""
my_cm = CM(all_data.A, all_data.b, constants.NBR_EPOCH, constants.NBR_UPDATES, constants.INITIAL_LEARNING_RATE, constants.RANK_S, constants.TIME_IN_SECONDS)
my_cm.train(True)
plt.plot(my_cm.loss_history, label='CM')
"""

"""
my_am = AM(all_data.A, all_data.b, constants.NBR_EPOCH, constants.NBR_UPDATES, constants.INITIAL_LEARNING_RATE, constants.RANK_S, constants.TIME_IN_SECONDS)
my_am.train(True)
plt.plot(my_am.loss_history, label='AM')
"""

"""
my_svrg = SVRG(all_data.A, all_data.b, constants.NBR_EPOCH, constants.NBR_UPDATES, constants.INITIAL_LEARNING_RATE, constants.TIME_IN_SECONDS)
my_svrg.train(True)
plt.plot(my_svrg.loss_history,  label='SVRG')
"""


"""
my_svrg2 = SVRG2(all_data.A, all_data.b, constants.NBR_EPOCH, constants.NBR_UPDATES, constants.INITIAL_LEARNING_RATE, constants.TIME_IN_SECONDS)
my_svrg2.train(False)
plt.plot(my_svrg2.loss_history, label='SVRG2')
"""

"""
plt.grid()
plt.yscale('log')
plt.show()
"""