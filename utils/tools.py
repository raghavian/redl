import numpy as np
import scipy.sparse as sp
import torch
from os.path import isfile
from os import rename
SMOOTH=1
import pdb
from sklearn.metrics import auc, roc_curve
import torch.nn.functional as F
import torch.nn as nn
from PIL.ImageFilter import GaussianBlur
EPS=1e-6
SMOOTH = 1
def wCELoss(prediction, target):
   w1 = 1.33  # False negative penalty
   w2 = .66  # False positive penalty
   return -torch.mean(w1 * target * torch.log(prediction.clamp_min(1e-3))
   + w2 * (1. - target) * torch.log(1. - prediction.clamp_max(.999)))

class GaussianFilter(object):
     """Apply Gaussian blur to the PIL image
     Args:
     sigma (float): Sigma of Gaussian kernel. Default value 1.0
     """
     def __init__(self, sigma=1):
          self.sigma = sigma
          self.filter = GaussianBlur(radius=sigma)
          
     def __call__(self, img):
          """
          Args:
               img (PIL Image): Image to be blurred.

          Returns:
               PIL Image: Blurred image.
          """
          return img.filter(self.filter)

     def __repr__(self):
          return self.__class__.__name__ + '(sigma={})'.format(self.sigma)

class GaussianLayer(nn.Module):
     def __init__(self):
          super(GaussianLayer, self, sigma=1, size=10).__init__()
          self.sigma = sigma
          self.size = size
          self.seq = nn.Sequential(
               nn.ReflectionPad2d(size), 
               nn.Conv2d(3, 3, size, stride=1, padding=0, bias=None, groups=3)
          )
          self.weights_init()

     def forward(self, x):
          return self.seq(x)

     def weights_init(self):
          s = self.size * 2 + 1
          k = np.zeros((s,s))
          k[s,s] = 1
          kernel = gaussian_filter(k,sigma=self.sigma)
          for name, f in self.named_parameters():
               f.data.copy_(torch.from_numpy(kernel))

class focalLoss(nn.Module):
     def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
          super(focalLoss, self).__init__()
          self.alpha = alpha
          self.gamma = gamma
          self.logits = logits
          self.reduce = reduce

     def forward(self, inputs, targets):
          if self.logits:
               BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
          else:
               BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
          pt = torch.exp(-BCE_loss)
          F_loss = self.alpha * ((1-pt)**self.gamma) * BCE_loss

          if self.reduce:
               return torch.mean(F_loss)
          else:
               return F_loss.sum()

class tverskyLoss(nn.Module):
    def __init__(self, alpha=0.7):
        super(tverskyLoss, self).__init__()
        self.alpha = alpha

    def forward(self, inputs, targets):
#        pdb.set_trace()
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        tp = (inputs * targets).sum()
        fn = ((1-inputs) * targets).sum()
        fp = (inputs * (1-targets)).sum()
        return 1-(tp+SMOOTH)/(tp+self.alpha*fn + (1-self.alpha)*fp + SMOOTH)


def computeAuc(target,preds):
     fpr, tpr, thresholds = roc_curve(target,preds)
     aucVal = auc(fpr,tpr)
     return aucVal

class hingeLoss(torch.nn.Module):

     def __init__(self):
          super(hingeLoss, self).__init__()

     def forward(self, output, target):
#          pdb.set_trace()
          target = 2*target-1
          output = 2*output-1
          hinge_loss = 1 - torch.mul(output, target)
          hinge_loss[hinge_loss < 0] = 0
          return hinge_loss.mean()


def makeBatchAdj(adj,bSize):

     E = adj._nnz()
     N = adj.shape[0]
     batch_idx = torch.zeros(2,bSize*E).type(torch.LongTensor)
     batch_val = torch.zeros(bSize*E)

     idx = adj._indices()
     vals = adj._values()

     for i in range(bSize):
          batch_idx[:,i*E:(i+1)*E] = idx + i*N
          batch_val[i*E:(i+1)*E] = vals

     return torch.sparse.FloatTensor(batch_idx,batch_val,(bSize*N,bSize*N))


def makeAdj(ngbrs, normalize=True):
     """ Create an adjacency matrix, given the neighbour indices
     Input: Nxd neighbourhood, where N is number of nodes
     Output: NxN sparse torch adjacency matrix 
     """
#     pdb.set_trace()
     N, d = ngbrs.shape
     validNgbrs = (ngbrs >= 0) # Mask for valid neighbours amongst the d-neighbours
     row = np.repeat(np.arange(N),d) # Row indices like in sparse matrix formats
     row = row[validNgbrs.reshape(-1)] #Remove non-neighbour row indices 
     col = (ngbrs*validNgbrs).reshape(-1) # Obtain nieghbour col indices
     col = col[validNgbrs.reshape(-1)] # Remove non-neighbour col indices
     data = np.ones(col.size)
     adj = sp.csr_matrix((np.ones(col.size, dtype=bool),(row, col)), shape=(N, N)).toarray() # Make adj matrix
     adj = adj + np.eye(N) # Self connections 
     adj = sp.csr_matrix(adj, dtype=np.float32)#/(d+1)
     if normalize:
          adj = row_normalize(adj)
     adj = sparse_mx_to_torch_sparse_tensor(adj) 
     
     return adj

def makeRegAdj(numNgbrs=26):
          """ Make regular pixel neighbourhoods"""
          idx = 0
          ngbrOffset = np.zeros((3,numNgbrs),dtype=int)
          for i in range(-1,2):
               for j in range(-1,2):
                    for k in range(-1,2):
                              if(i | j | k):
                                     ngbrOffset[:,idx] = [i,j,k]
                                     idx+=1
          idx = 0
          ngbrs = np.zeros((numEl, numNgbrs), dtype=int)

          for i in range(xdim):
                    for j in range(ydim):
                              for k in range(zdim):
                                     xIdx = np.mod(ngbrOffset[0,:]+i,xdim)
                                     yIdx = np.mod(ngbrOffset[1,:]+j,ydim)
                                     zIdx = np.mod(ngbrOffset[2,:]+k,zdim)
                                     ngbrs[idx,:] = idxVol[xIdx, yIdx, zIdx]
                                     idx += 1


def makeAdjWithInvNgbrs(ngbrs, normalize=False):
     """ Create an adjacency matrix, given the neighbour indices including invalid indices where self connections are added.
     Input: Nxd neighbourhood, where N is number of nodes
     Output: NxN sparse torch adjacency matrix 
     """
     np.random.seed(2)
#     pdb.set_trace()
     N, d = ngbrs.shape
     row = np.arange(N).reshape(-1,1)
     random = np.random.randint(0,N-1,(N,d))
     valIdx = np.array((ngbrs < 0),dtype=int)
     ngbrs = random*valIdx + ngbrs*(1-valIdx)# Mask for valid neighbours amongst the d-neighbours
     row = np.repeat(row,d).reshape(-1) # Row indices like in sparse matrix formats
     col = ngbrs.reshape(-1) # Obtain nieghbour col indices
     data = np.ones(col.size)
     adj = sp.csr_matrix((np.ones(col.size, dtype=bool),(row, col)), shape=(N, N)).toarray() # Make adj matrix
     adj = adj + np.eye(N) # Self connections 
     adj = sp.csr_matrix(adj, dtype=np.float32)#/(d+1)
     if normalize:
          adj = row_normalize(adj)
     adj = sparse_mx_to_torch_sparse_tensor(adj) 
     adj = adj.coalesce()
     adj._values = adj.values()     
     return adj


def transformers(adj):
     """ Obtain source and sink node transformer matrices"""
     edges = adj._indices()
     N = adj.shape[0]
     nnz = adj._nnz()
     val = torch.ones(nnz)
     idx0 = torch.arange(nnz)

     idx = torch.stack((idx0,edges[1,:]))
     n2e_in = torch.sparse.FloatTensor(idx,val,(nnz,N))

     idx = torch.stack((idx0,edges[0,:]))
     n2e_out = torch.sparse.FloatTensor(idx,val,(nnz,N))

     return n2e_in, n2e_out

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
     """Convert a scipy sparse matrix to a torch sparse tensor."""
     sparse_mx = sparse_mx.tocoo().astype(np.float32)
     indices = torch.from_numpy(np.vstack((sparse_mx.row,
                                                 sparse_mx.col))).long()
     values = torch.from_numpy(sparse_mx.data)
     shape = torch.Size(sparse_mx.shape)
     return torch.sparse.FloatTensor(indices, values, shape)

def to_linear_idx(x_idx, y_idx, num_cols):
     assert num_cols > np.max(x_idx)
     x_idx = np.array(x_idx, dtype=np.int32)
     y_idx = np.array(y_idx, dtype=np.int32)
     return y_idx * num_cols + x_idx


def row_normalize(mx):
     """Row-normalize sparse matrix"""
     rowsum = np.array(mx.sum(1), dtype=np.float32)
     r_inv = np.power(rowsum, -1).flatten()
     r_inv[np.isinf(r_inv)] = 0.
     r_mat_inv = sp.diags(r_inv)
     mx = r_mat_inv.dot(mx)
     return mx

def to_2d_idx(idx, num_cols):
     idx = np.array(idx, dtype=np.int64)
     y_idx = np.array(np.floor(idx / float(num_cols)), dtype=np.int64)
     x_idx = idx % num_cols
     return x_idx, y_idx

#def dice_loss(preds, labels,reduce=False):
#     "Return dice score. "
#     pdb.set_trace()
#     preds = preds.view(-1)
#     labels = labels.view(-1)
#     preds_sq = preds**2
#     loss = 1 - (2. * ((preds * labels).sum()) + EPS) / \
#           (preds_sq + labels + EPS).sum()
#     if reduce:
#        return loss.sum()/loss.shape[0]
#     else:
#        return loss.sum()

class dice_loss(nn.Module):
    def __init__(self):
        super(dice_loss, self).__init__()

    def forward(self, inputs, targets):
#        pdb.set_trace()
        SMOOTH = 1e-6
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        return 1-(2. * (inputs * targets).sum() + SMOOTH)/ \
                  (inputs**2 + targets**2 + SMOOTH).sum()
        tp = (inputs * targets).sum()
        fn = ((1-inputs) * targets).sum()
        fp = (inputs * (1-targets)).sum()
        return 1-(tp+SMOOTH)/(tp+self.alpha*fn + (1-self.alpha)*fp + SMOOTH)

       

def binary_accuracy(labels,output):
#     pdb.set_trace()
#     b = labels.shape[0]
     labels = labels.view(-1)
     preds = (output > 0.5).view(-1)
     correct = preds.type_as(labels).eq(labels).double()
     correct = correct.sum()
     return correct / len(labels)

def iou_acc(labels,output):
     pdb.set_trace()
#     b = labels.shape[0]
     preds = (output > 0.5).type_as(labels)
     inter = (preds & labels).float().sum(1,2)
     union = (preds | labels).float().sum(1,2)
     acc = (inter+EPS) / (union+EPS)
     return acc.mean()


def balanced_accuracy(output, labels):
#    pdb.set_trace()
     preds = output > 0.5
     posIdx = (labels==1)
     posCorrect = preds[posIdx].type_as(labels).eq(labels[posIdx]).double()
     posAcc = (len(posCorrect)+EPS)/(len(labels[posIdx])+EPS) 
     negCorrect = preds[~posIdx].type_as(labels).eq(labels[~posIdx]).double()
     negAcc = (len(negCorrect)+EPS)/(len(labels[~posIdx])+EPS)
     
     return (posAcc+negAcc)/2

def mad(preds,labels,maxAge=228,minAge=1):
#    pdb.set_trace()
     diff = maxAge-minAge
     labels = np.round(labels*diff+minAge)
     preds = np.round(preds*diff+minAge)
     return np.mean(np.abs(preds -labels))

def multiClassAccuracy(output, labels):
     pdb.set_trace()
     preds = output.argmax(1)
#     preds = (output > (1.0/labels.shape[1])).type_as(labels)
     correct = (preds == labels.view(-1))
     correct = correct.sum().float()
     return correct / len(labels)

def regrAcc(output, labels):
#     pdb.set_trace()
     preds = output.round().type(torch.long).type_as(labels)
#     preds = (output > (1.0/labels.shape[1])).type_as(labels)
     correct = (preds == labels.view(-1))
     correct = correct.sum().float()
     return correct / len(labels)


def rescaledRegAcc(output,labels,lRange=37,lMin=-20):
#     pdb.set_trace()
     preds = (output+1)*(lRange)/2 + lMin
     preds = preds.round().type(torch.long).type_as(labels)
#     preds = (output > (1.0/labels.shape[1])).type_as(labels)
     correct = (preds == labels.view(-1))
     correct = correct.sum().float()
     return correct / len(labels)

def focalCE(preds, labels, gamma=1):
     "Return focal cross entropy"
     loss = -torch.mean( ( ((1-preds)**gamma) * labels * torch.log(preds) ) \
     + ( ((preds)**gamma) * (1-labels) * torch.log(1-preds) ) )
     return loss

def dice(labels,preds,test=False):
     b = labels.shape[0]
     "Return dice score"
     labels = labels.view(b,-1)
     preds = preds.view(b,-1)
     preds_bin = (preds > 0.5).type_as(labels)
     acc = 2. * (torch.sum(preds_bin * labels,dim=1) + EPS)/ (preds_bin.sum(1) + labels.sum(1)+EPS)
     if test:
        return acc.mean(), acc
     else:
        return acc.mean()

def wBCE(preds, labels, w):
     "Return weighted CE loss."
     return -torch.mean( w*labels*torch.log(preds) + (1-w)*(1-labels)*torch.log(1-preds) )

def makeLogFile(filename="lossHistory.txt"):
     if isfile(filename):
          rename(filename,"lossHistoryOld.txt")

     with open(filename,"w") as text_file:
          print('Epoch,lossTr,accTr,lossVl,accVl,lossTs,accTs,time(s)',file=text_file)
     print("Log file created...")
     return

def writeLog(logFile, epoch, lossTr, accTr, lossVl, accVl, lossTs, accTs, eTime):
     print('Epoch:{:04d},'.format(epoch + 1),
            'lossTr:{:.4f},'.format(lossTr),
            'accTr:{:.4f},'.format(accTr),
            'lossVl:{:.4f},'.format(lossVl),
            'accVl:{:.4f},'.format(accVl),
            'lossTs:{:.4f},'.format(lossTs),
            'accTs:{:.4f},'.format(accTs),
            'time:{:.4f}'.format(eTime))

     with open(logFile,"a") as text_file:
         print('Epoch:{:04d},'.format(epoch + 1),
                'lossTr:{:.4f},'.format(lossTr),
                'accTr:{:.4f},'.format(accTr),
                'lossVl:{:.4f},'.format(lossVl),
                'accVl:{:.4f},'.format(accVl),
                'lossTs:{:.4f},'.format(lossTs),
                'accTs:{:.4f},'.format(accTs),
                'time:{:.4f}'.format(eTime),file=text_file)

     return

def plotLearningCurve():
          plt.clf()
          tmp = np.load('loss_tr.npz')['arr_0']
          plt.plot(tmp,label='Tr.Loss')
          tmp = np.load('loss_vl.npz')['arr_0']
          plt.plot(tmp,label='Vl.Loss')
          tmp = np.load('dice_tr.npz')['arr_0']
          plt.plot(tmp,label='Tr.Dice')
          tmp = np.load('dice_vl.npz')['arr_0']
          plt.plot(tmp,label='Vl.Dice')
          plt.legend()
          plt.grid()
          plt.show()
