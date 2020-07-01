from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


from Gaussian import GaussianDistribution


class BayesianNRegression:
    #self.mapping  a function could map x->a
    #self.Xdim 
    #self.Adim
    #self.A_gsd gaussian distribution for A space
    def __init__(self, mapping, miu=None,uncertainty=1):
        self.mapping = mapping
        self.Adim = mapping.Adim
        self.Xdim = mapping.Xdim
        self.A_gsd = GaussianDistribution(dim=self.Adim,miu=miu,sigma=np.diag([uncertainty**2]*self.Adim))
        self.nodes = []
    def GetNode(self, *keys, belief=3,display=False):
        keys = np.array(keys)
        if len(keys.shape) != 1:
            raise Exception("only get 1 node at a time")
        if len(keys) != self.Xdim:
            raise Exception("dimension not fitted")
        self.nodes.append(keys)

        a_vec, dist = self.mapping.GetVector(keys)

        base_vec = a_vec / np.linalg.norm(a_vec)

        #build R
        R = [base_vec]
        index=0
        count=0
        while count < self.Adim - 1:
            current_vec = np.zeros(self.Adim)
            current_vec[index] = 1
            index += 1
            if current_vec.tolist() == base_vec.tolist():
                continue
            else:
                count+=1
            for vec in R:
                current_vec -= vec * np.sum(vec * current_vec)
            current_vec = current_vec / np.linalg.norm(current_vec)    
            R.append(current_vec)
        R = np.stack(R,-1)
        R = R[:, ::-1]
        R[:,0]*R[:,1]

        #build S
        belief = np.exp(belief)
        sdia = [belief] * (self.Adim - 1) + [1. / belief]
        S = np.diag(sdia)

        #build sigma
        T = R.dot(S)
        sigma = T.dot(T.T)

        #build miu
        miu = base_vec * dist
        
        #build gaussian
        node_gsd = GaussianDistribution(dim=self.Adim, miu=miu, sigma=sigma)
        
        if display:
            node_gsd.Plot(scale=[-2,2],title='Node{},{}'.format(keys[0],keys[1]))

        #mul new gaussian
        self.A_gsd = self.A_gsd.Mul(node_gsd)
    def Integrate(self, xdensity=100, adensity=100, xscale=[-2, 2], ascale=[-2, 2]):

        aslice_len = adensity ** (self.Adim)
        xslice_len = int(5e8 /aslice_len)
        if xslice_len < 1:
            xslice_len = 1

        step = (xscale[1]-xscale[0])/xdensity
        x_s = np.arange(xscale[0], xscale[1], step)
        x_pos = None
        loc = {'x_pos':x_pos,'x_s':x_s,'np':np}
        glb = {}
        exec('x_pos = np.meshgrid({})'.format(''.join('x_s,' for i in range(self.Xdim))[:-1]),glb,loc) 
        x_pos = loc['x_pos']
        x_pos = np.stack(x_pos)
        x_pos = x_pos.reshape([self.Xdim,-1])
        step = (ascale[1]-ascale[0]) / adensity
        a_s = np.arange(ascale[0],ascale[1],step)
        a_pos = None
        loc = {'a_pos':a_pos,'a_s':a_s,'np':np}
        glb = {}
        exec('a_pos = np.meshgrid({})'.format(''.join('a_s,' for i in range(self.Adim-1))[:-1]),glb,loc)
        a_pos = loc['a_pos']
        a_pos = np.stack(a_pos)
        a_pos = a_pos.reshape([self.Adim-1,-1])
        x_pos = torch.FloatTensor(x_pos).cuda()
        a_pos = np.ascontiguousarray(a_pos,np.float32)
        a_pos = torch.FloatTensor(a_pos).cuda()

        if not self.mapping.pole:
            #xpos shape [xdim,xnum]
            #apos shape [adim-1,anum]

            a_tile = a_pos.reshape([ a_pos.shape[0], a_pos.shape[1],1])
            
            lprob=[]
            for i in tqdm(np.arange(0,x_pos.shape[1],xslice_len)):
                x_slice = x_pos[:, i: (i + xslice_len)]
                a_tiled = a_tile.repeat(1, 1, x_slice.shape[-1])
                a_full, delta = self.mapping.GetFullA(x_s=x_slice, a_s=a_tiled, step=step)  #delta.shape [xnum]
                a_full = a_full.reshape([self.Adim, -1])  #a_full shape [adim, anum*xnum]
                prob = self.A_gsd[a_full]
                prob = prob.reshape([a_pos.shape[1],x_slice.shape[-1]])
                prob *= delta.cpu().numpy()
                prob = np.sum(prob, 0)
                lprob.append(prob.copy())
            probs = np.concatenate(lprob)
            prob = probs.reshape([xdensity] * self.Xdim)


    
        else:
            prob = []
            for x in tqdm(x_pos.T):
                full_a, delta = self.mapping.GetFullA(x, a_pos,step=step)
                lprob = self.A_gsd[full_a]
                prob.append(np.sum(lprob)*delta)
                
            prob = np.array(prob)
            prob = prob.reshape([xdensity]*self.Xdim)
        # prob = prob.T
        return prob
    def PlotX(self, read=False,show_nodes=False, style='2D',choice_num=1,xdensity=100, adensity=100, xscale=[-2, 2], ascale=[-2, 2]):
        #style choose from ['2D','3D','choice2D','choice3D']

        if self.Xdim!=2 and style!='choice3D':
            raise Exception("xdim is not 2 but {}".format(self.Xdim))
        elif self.Xdim != 3 and style == 'choice3D':
            raise Exception("choice 3D plot 3D lines")
        
        if style == 'choice2D':
            step = (xscale[1]-xscale[0]) / xdensity
            x_s = np.arange(xscale[0],xscale[1],step)
            a_s = self.A_gsd.miu
            x_s = self.mapping.GetFullX(x_s=x_s,a_s=a_s)
            if type(x_s) != type(None):
                plt.plot(x_s[0],x_s[1])
            else:
                return 
            if show_nodes and len(self.nodes)>0:
                nodes = np.array(self.nodes)
                sizes = [30+10*i for i in range(len(nodes))]
                plt.scatter(nodes[:,0], nodes[:,1], c='none', edgecolors='c', s=sizes, marker='d')

        elif style == 'choice3D':
            pass
        elif style == '2D':
            if not read:
                x_prob = self.Integrate( xdensity=xdensity,
                                    adensity=adensity,
                                    xscale=xscale,
                                    ascale=ascale)
            else:
                x_prob = self.Load()
            f, ax = plt.subplots(figsize=(10, 7))
            sns.heatmap(x_prob, fmt='d', cmap='Spectral_r',ax=ax)
            ax.invert_yaxis()
            new_axis = []
            step = (xscale[1] - xscale[0]) / xdensity
            x = np.arange(xscale[0],xscale[1],step)
            step = int(xdensity/5)
            for i_iter,i in enumerate(x):
                if i_iter % step == 0 or i_iter == xdensity-1:
                    new_axis.append(round(x[i_iter], 3))
                else:
                    new_axis.append('')
            plt.title('X space')
            plt.xlabel('x0')
            plt.ylabel('x1')
            plt.xticks(np.arange(xdensity),new_axis)
            plt.yticks(np.arange(xdensity),new_axis)
            if show_nodes and len(self.nodes)>0:
                nodes = np.array(self.nodes)
                nodex = [((i-xscale[0])*xdensity)/(xscale[1]-xscale[0]) for i in nodes[:,0]]
                nodey = [((i-xscale[0])*xdensity)/(xscale[1]-xscale[0]) for i in nodes[:,1]]
                sizes = [30+10*i for i in range(len(nodes))]
                plt.scatter(nodex, nodey, c='none', edgecolors='c', s=sizes, marker='d')
        elif style == '3D':
            x_prob = self.Integrate( xdensity=xdensity,
                                adensity=adensity,
                                xscale=xscale,
                                ascale=ascale)
            x_prob = x_prob[::-1]
            step = (xscale[1]-xscale[0])/xdensity
            x = np.arange(xscale[0],xscale[1],step)
            x,y = np.meshgrid(x,x)
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.plot_surface(x, y, x_prob, rstride=1, cstride=1, cmap='rainbow')
            
            if show_nodes and len(self.nodes) > 0:
                nodes = np.array(self.nodes)
                nodex = [int(((i-xscale[0])*xdensity)/(xscale[1]-xscale[0]))-1 for i in nodes[:,0]]
                nodey = [int(((i - xscale[0]) * xdensity) / (xscale[1] - xscale[0]))-1 for i in nodes[:, 1]]
                ax.scatter(nodes[:, 0], nodes[:, 1], 0, color='r')
                ax.invert_yaxis()
            
            plt.xlabel('x0',size = 10)
            plt.ylabel('x1',size = 10)
            ax.set_zlabel('probability',size=10)
        else:
            raise Exception("Do not have this type {} only '2D' and '3D' ".format(style))
        return plt
  