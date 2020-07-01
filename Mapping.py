import numpy as np
import torch 


class LinearExp1:# x1 * a1 + x2 * a2 + a3= 0
    def __init__(self):       
        self.Xdim = 2
        self.Adim = 3
        self.pole=False
    def _Item(self,x_s):
        return torch.stack([x_s[0],x_s[1]])
    def GetFullA(self, x_s, a_s,step,*useless):
        # input x_s tensor shape [xdim,xnum]
        #       a_s tensor shape [adim-1, anum, xnum]
        # output full_a tensor shape [adim, anum, xnum]
        #       delta tensor shape [xnum]
        items = self._Item(x_s)
        delta = torch.sqrt(torch.sum(items**2,0)+1)*(step**(self.Adim-1))
        items = items.reshape([items.shape[0],1,items.shape[1]])
        last_a = -torch.sum(items*a_s,0)
        last_a = last_a.reshape([1] + list(last_a.shape))
        last_a = last_a.float()
        full_a = torch.cat([a_s,last_a],0)
        return full_a, delta
    def GetVector(self, x_s):
        # input list of num
        dist=0
        x_s = torch.FloatTensor(x_s)
        if len(x_s.shape) == 1:
            one = torch.FloatTensor([1])
        else:
            one = torch.FloatTensor([1]*x_s.shape[-1])
        vec = torch.cat([x_s,one])
        return vec.numpy(),dist
    def GetFullX(self,x_s,a_s):
        if a_s.any() == 0:
            print("all zero ??")
            return np.array([0,0])
        elif a_s[1]==0:   
            y_s = x_s
            x_s = -a_s[2]/a_s[0]
            x_s = np.stack([x_s,y_s])
        else:
            x_s = np.stack([x_s,(-a_s[2]-a_s[0]*x_s)/(a_s[1])])
        return x_s

class UnLinearExp: #x1 = a0*x0^3 + a1*x0^2 + a2*x0 + a3;
    def __init__(self):
        self.Xdim = 2
        self.Adim = 4
        self.pole = False
    def _Item(self, x_s):
        # [items, xnum]
        return torch.stack([x_s[0] * 3,  x_s[0] ** 2,  x_s[0]])
    def GetFullA(self, x_s, a_s, step):
        items = self._Item(x_s)
        delta = torch.sqrt(torch.sum(items ** 2, 0) +1) * (step ** (self.Adim - 1))
        items = items.reshape([items.shape[0], 1, items.shape[1]])
        last_a = -torch.sum(a_s * items, 0) + x_s[1]
        last_a = last_a.reshape([1] + list(last_a.shape))
        last_a = last_a.float()
        full_a = torch.cat([a_s, last_a], 0)
        return full_a, delta
    def GetVector(self, x_s):
        x_s = torch.FloatTensor(x_s)
        items = self._Item(x_s)
        dist = x_s[1].abs() / torch.sqrt(torch.sum(items ** 2) + 1)
        one = torch.FloatTensor([1])
        vec = torch.cat([items,one])
        return vec.numpy(), dist.item()
    def GetFullX(self, x_s, a_s):
        x_s = torch.tensor(x_s)
        a_s = torch.tensor(a_s)
        items = self._Item(x_s.reshape([1,-1]))
        last_x = torch.sum(items * a_s[:-1].reshape([-1,1]), 0) + a_s[-1]
        x_s = torch.stack([x_s, last_x])
        return x_s.numpy()