from BayesNRegression import BayesNRegression
from mapping import *
import random



def GetData():
    x = np.arange(-2, 2, 0.1)
    random.shuffle(x)
    y = x ** 2 - x - 1
    return np.stack([x,y])


if __name__ == '__main__':
    f =UnLinearExp()
    bnr = BayesianNRegression(f,uncertainty=2)
    data = GetData()
    for x in data:
        bnr.GetNode(x)
    bnr.PlotX(style='choice2D', xscale=[-3, 3])
    bnr.PlotX(style='2D', xscale=[-3, 3])
    bnr.PlotX(style='3D',xscale=[-3,3])
