
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")

from temSim import Simulator

filepath = '../data/AAPL_2024_03_06_1D_stock_raw_v5_best.csv' #stock
sim = Simulator(filepath)
sim.run()



