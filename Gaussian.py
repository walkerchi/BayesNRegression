import numpy as np
import torch
import torch.nn as nn
import seaborn as sns
from matplotlib import pyplot as plt

class GaussianDistribution:
    def __init__(self, dim=1, miu=None, sigma=None):
        # miu could be float and array or list
        #sigma could be float and array or list
        def check(name, var,shape):
            if type(var) != np.ndarray:
                raise Exception("{} type error,must be array or list of shape{},but got {}".format(name, shape, type(var)))
            elif list(var.shape) != list(shape):
                raise Exception("{} shape error,must be shape of{} but got shape{}".format(name,shape,var.shape))

        self.dim = dim
        if dim == 1:
            if miu == None:
                self.miu = 0.
            else:
                self.miu = miu
            if sigma == None:
                self.sig = 0.33
            else:
                self.sig = sigma
        else:
            if miu is None:
                miu = np.zeros((dim))
            elif type(miu) == list:
                miu = np.array(miu)
            if sigma is None:
                sigma = np.eye((dim)) / 3
            elif type(sigma) == list:
                sigma = np.array(sigma)
            check('miu', miu, [dim])
            check('sigma', sigma, [dim,dim])
            self.miu = miu
            self.sig = sigma
            self._Update()
            # self.cov_ = torch.FloatTensor(np.linalg.inv(sigma)).cuda()
            # covdet = torch.tensor(np.linalg.det(sigma))
            # self.norm = (torch.sqrt( covdet )*(np.sqrt(2*np.pi)**self.dim)).cuda()   
    def __getitem__(self, *keys):
        keys = keys[0]
        if len(keys) < self.dim:
            raise Exception("Key erorr, got {} keys but self dim is {}".format(len(keys), self.dim))
            
        if type(keys) != torch.Tensor:
            x = torch.FloatTensor(keys).cuda()
        else:
            x = keys

        if self.dim == 1:
            return np.exp(-(x - self.miu) ** 2 / (2 * self.sig ** 2)) / np.sqrt(2 * np.pi * self.sig ** 2)
        else:
            x = x.T #x.shape([num,dim])
            miu = torch.FloatTensor(self.miu).cuda()
            res = x - miu
            exponent = torch.sum( (torch.mm(res, self.cov_) * res),-1)
            prob = torch.exp(-exponent/2) / self.norm
            return prob.cpu().numpy()
    def _Update(self):
        self.cov_ = torch.FloatTensor(np.linalg.inv(self.sig)).cuda()
        covdet = torch.tensor(np.linalg.det(self.sig))
        self.norm = (torch.sqrt( covdet )*(np.sqrt(2*np.pi)**self.dim)).cuda()   
    def Mul(self, gsd):
        if type(gsd) != GaussianDistribution:
            raise Exception("Not GaussianDistribution type:{}".format(type(gsd)))
        if self.dim != gsd.dim:
            raise Exception("Dim not fit {} and {}".format(self.dim, gsd.dim))
        out = GaussianDistribution(dim=self.dim)
        if self.dim == 1:
            out.miu = (self.miu * gsd.sig ** 2 + gsd.miu * self.sig ** 2) / (self.sig ** 2 + gsd.sig ** 2)
            out.sig = (self.sig ** 2 * gsd.sig ** 2) / (self.sig ** 2 + gsd.sig ** 2)
        else:
            # sig = (cov1^-1+cov2^-1)^-1
            # miu = sig*cov1^-1*miu1 + sig*cov2^-1*miu2
            cov1_ = np.linalg.inv(self.sig)
            cov2_ = np.linalg.inv(gsd.sig)
            out.sig = np.linalg.inv(cov1_ + cov2_)
            out.miu = out.sig.dot(cov1_).dot(self.miu) + out.sig.dot(cov2_).dot(gsd.miu)
            out._Update()
            # out.miu = np.dot(out.sig,cov1_,self.miu) + np.dot(out.sig,cov2_,gsd.miu)
        return out
    def Add(self, gsd):
        if type(gsd) != GaussianDistribution:
            raise Exception("Not GaussianDistribution type:{}".format(type(gsd)))
        if self.dim != gsd.dim:
            raise Exception("Dim not fit {} and {}".format(self.dim, gsd.dim))
        out = GaussianDistribution(dim=self.dim)
        out.miu = self.miu + gsd.miu
        out.sig = self.sig + gsd.sig
        out._Update()
        return out
    def Sub(self, gsd):
        if type(gsd) != GaussianDistribution:
            raise Exception("Not GaussianDistribution type:{}".format(type(gsd)))
        if self.dim != gsd.dim:
            raise Exception("Dim not fit {} and {}".format(self.dim, gsd.dim))
        out = GaussianDistribution(dim=self.dim)
        out.miu = self.miu - gsd.miu
        out.sig = self.sig + gsd.sig
        out._Update()
        return out
    def Plot(self, style='2D',scale=None, density=100,title='GaussianDistribution',xlabel='',ylabel=''):
        if scale is None:
            if self.dim == 1:
                scale = [-3 * (self.sig+self.miu), 3 * (self.sig+self.miu)]
            else:
                scale = [-3 * (np.max(self.sig.diagonal())+np.max(self.miu)), 3 * (np.max(self.sig.diagonal())+np.max(self.miu))]
        if len(scale) != 2:
            raise Exception("scale must be len 2 but got{}".format(len(scale)))
 

        step = (scale[1] - scale[0]) / density
        x = np.arange(scale[0],scale[1],step)

        if self.dim == 1:
            y = self[x]
            plt.figure()
            plt.plot(x, y)
            plt.show()
        elif self.dim == 2:
            x_tile = np.tile(x.reshape([x.shape[0], 1]), [1, density])
            y_tile = x_tile.T
            z = self[x_tile, y_tile]
            if style=='2D':
                f, ax = plt.subplots(figsize = (10, 7))
                sns.heatmap(z, fmt='d', cmap='Spectral_r',ax=ax)
                # plt.setp(ax.get_y_ticklabels(), rotation=360, horizontalalignment='right')
                ax.invert_yaxis()
                new_axis = []
                step = int(len(x)/self.axis_len)
                for i_iter,i in enumerate(x):
                    if i_iter % step == 0 or i_iter == len(x)-1:
                        new_axis.append(round(x[i_iter], 3))
                    else:
                        new_axis.append('')

                plt.xticks(np.arange(density),new_axis)
                plt.yticks(np.arange(density),new_axis)
            elif style == '3D':
                x,y = np.meshgrid(x,x)
                fig = plt.figure()
                ax = Axes3D(fig)
                ax.plot_surface(x,y,z,rstride=1,cstride=1,cmap='rainbow')
                ax.set_zlabel('probability', size=10)
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.show()
        else:
            print("Dimension are too high that it's hard to figure out the plot")
        plt.show()     
    def Generate2D(self, angle=0, strech=[1,1], shift=[0,0] ,atype='angle'):
        #angle is for degree float
        #strech list of array
        self.dim = 2
        if len(strech) != 2:
            raise Exception("Shift len error must be 2 but got {}".format(len(shift)))
        if len(strech) != 2:
            raise Exception("Strech len error must be 2 but got {}".format(len(strech)))
        if atype == 'angle':
            angle = angle * np.pi / 180
            cos = np.cos(angle)
            sin = np.sin(angle)
        elif atype == 'tan':
            cos = np.sqrt(1 / (1 + angle ** 2))
            sin = cos * angle

        rotate = np.array([[cos, -sin], [sin, cos]])
        strech = np.diag(strech)
        cov = rotate.dot(strech)
        cov = cov.dot(cov.T)
        self.sig = cov
        self.miu = np.array(shift)

  
