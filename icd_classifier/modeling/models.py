from gensim.models import KeyedVectors
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.nn.init import xavier_uniform
from torch.nn.init import xavier_uniform_ as xavier_uniform
from torch.autograd import Variable
import numpy as np

from icd_classifier.data import extract_wvs
from icd_classifier.settings import MIMIC_3_DIR


class BaseModel(nn.Module):

    def __init__(
            self, Y, embed_file, dicts, lmbda=0,
            dropout=0.5, gpu=True, embed_size=100):
        super(BaseModel, self).__init__()
        torch.manual_seed(1337)
        self.gpu = gpu
        self.Y = Y
        self.embed_size = embed_size
        self.embed_drop = nn.Dropout(p=dropout)
        self.lmbda = lmbda

        # make embedding layer
        if embed_file:
            print("loading pretrained embeddings...")
            W = torch.Tensor(extract_wvs.load_embeddings(embed_file))

            self.embed = nn.Embedding(W.size()[0], W.size()[1], padding_idx=0)
            self.embed.weight.data = W.clone()
        else:
            # add 2 to include UNK and PAD
            vocab_size = len(dicts['ind2w'])
            self.embed = nn.Embedding(vocab_size+2, embed_size, padding_idx=0)

    def _get_loss(self, yhat, target, diffs=None):
        # calculate the BCE
        loss = F.binary_cross_entropy_with_logits(yhat, target)

        # add description regularization loss if relevant
        if self.lmbda > 0 and diffs is not None:
            diff = torch.stack(diffs).mean()
            loss = loss + diff
        return loss

    def embed_descriptions(self, desc_data, gpu):
        # label description embedding via convolutional layer
        # number of labels is inconsistent across instances,
        # so have to iterate over the batch
        b_batch = []
        for inst in desc_data:
            if len(inst) > 0:
                if gpu:
                    lt = Variable(torch.cuda.LongTensor(inst))
                else:
                    lt = Variable(torch.LongTensor(inst))
                d = self.desc_embedding(lt)
                d = d.transpose(1, 2)
                d = self.label_conv(d)
                d = F.max_pool1d(torch.tanh(d), kernel_size=d.size()[2])
                d = d.squeeze(2)
                b_inst = self.label_fc1(d)
                b_batch.append(b_inst)
            else:
                b_batch.append([])
        return b_batch

    def _compare_label_embeddings(self, target, b_batch, desc_data):
        # description regularization loss 
        # b is the embedding from description conv
        # iterate over batch because each instance has different # labels
        diffs = []
        for i, bi in enumerate(b_batch):
            ti = target[i]
            inds = torch.nonzero(ti.data).squeeze().cpu().numpy()

            zi = self.final.weight[inds, :]
            diff = (zi - bi).mul(zi - bi).mean()

            # multiply by number of labels to make sure overall mean
            # is balanced with regard to number of labels
            diffs.append(self.lmbda*diff*bi.size()[0])
        return diffs


class BOWPool(BaseModel):
    """
        Logistic regression model over average or max-pooled word vector input
    """

    def __init__(
            self, Y, embed_file, lmbda, gpu, dicts,
            pool='max', embed_size=100, dropout=0.5, code_emb=None):
        super(BOWPool, self).__init__(
            Y, embed_file, dicts, lmbda,
            dropout=dropout, gpu=gpu, embed_size=embed_size)
        self.final = nn.Linear(embed_size, Y)
        # for nn.Linear see https://pytorch.org/.../torch.nn.Linear
        # the embed_size and Y define the weight matrix size.
        if code_emb:
            self._code_emb_init(code_emb, dicts)
        else:
            xavier_uniform(self.final.weight)
        self.pool = pool
    
    # initialisation of the weight size as the code embeddings. -HD
    def _code_emb_init(self, code_emb, dicts):
        code_embs = KeyedVectors.load_word2vec_format(code_emb)
        # classmethod load_word2vec_format(fname, fvocab=None, binary=False,
        #   encoding='utf8', unicode_errors='strict',
        #   limit=None, datatype=<class 'numpy.float32'>)
        # Load the input-hidden weight matrix from the original
        #   C word2vec-tool format.
        weights = np.zeros(self.final.weight.size())
        for i in range(self.Y):
            code = dicts['ind2c'][i]
            weights[i] = code_embs[code]
        # set weight as the code embeddings.
        self.final.weight.data = torch.Tensor(weights).clone()

    def forward(self, x, target, desc_data=None, get_attention=False):
        # get embeddings
        x = self.embed(x)
        
        if self.pool == 'max':
            x = F.max_pool1d(x)
        else:
            # average pooling
            x = torch.mean(x, 1)
        logits = F.sigmoid(self.final(x))
        # only using the pooled, document embedding for logistic regression.
        # In this case, it is also possible to apply SVM for the task.
        loss = self._get_loss(logits, target, diffs=desc_data)
        return logits, loss, None
