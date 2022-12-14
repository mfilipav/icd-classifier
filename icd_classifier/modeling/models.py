from gensim.models import KeyedVectors
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# from torch.nn.init import xavier_uniform
from torch.nn.init import xavier_uniform_ as xavier_uniform
from math import floor
import logging
from icd_classifier.data import extract_wvs


class BaseModel(nn.Module):

    def __init__(
            self, number_labels, embeddings_file, dicts, lmbda=0,
            dropout=0.5, gpu=True, embedding_size=100):
        super(BaseModel, self).__init__()
        torch.manual_seed(1337)
        self.gpu = gpu
        self.number_labels = number_labels
        self.embedding_size = embedding_size
        # TODO: do we really need embedding dropout?
        self.embed_drop = nn.Dropout(p=dropout)
        self.lmbda = lmbda

        # make embedding layer
        if embeddings_file:
            W = torch.Tensor(extract_wvs.load_embeddings(embeddings_file))
            # https://discuss.pytorch.org/t/how-does-nn-embedding-work/88518/3
            # https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
            # shape: (num_embeddings, embedding_dim)
            num_embeddings = W.size()[0]
            embedding_dim = W.size()[1]
            self.embed = nn.Embedding(
                num_embeddings=num_embeddings, embedding_dim=embedding_dim,
                padding_idx=0)
            self.embed.weight.data = W.clone()
            logging.info("Loaded pretrained embeddings from: {}".format(
                embeddings_file))
            logging.info(
                "size of the dictionary of embeddings: {}, "
                "size of each embedding vector: {}".format(
                        num_embeddings, embedding_dim))
        else:
            # add 2 to vocab size to include UNK and PAD
            vocab_size = len(dicts['ind2w'])
            num_embeddings = vocab_size + 2
            self.embed = nn.Embedding(
                num_embeddings=num_embeddings, embedding_dim=embedding_size,
                padding_idx=0)
            logging.info(
                "No pretrained embeddings file, initialized new embedding "
                "with 2 extra tokens for UNK and PAD, besides vocab of "
                "size: {}".format(vocab_size))
            logging.info(
                "size of the dictionary of embeddings: {}, "
                "size of each embedding vector: {}".format(
                        num_embeddings, embedding_size))

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


class LogReg(BaseModel):
    """
        Logistic regression model over average or max-pooled word vector input
        TODO: remove 'get_attention' -- refactor from train --
            models(desc_data and get_attention should be kwargs)
    """

    def __init__(
            self, number_labels, embeddings_file, lmbda, gpu, dicts,
            pool='max', embedding_size=100, dropout=0.5,
            codes_embeddings=None):
        super(LogReg, self).__init__(
            number_labels, embeddings_file, dicts, lmbda,
            dropout=dropout, gpu=gpu, embedding_size=embedding_size)
        self.final = nn.Linear(embedding_size, number_labels)
        # for nn.Linear see https://pytorch.org/.../torch.nn.Linear
        # the embedding_size and number_labels define the weight matrix size.
        if codes_embeddings:
            self._codes_embeddings_init(codes_embeddings, dicts)
        else:
            xavier_uniform(self.final.weight)
        self.pool = pool

    # initialisation of the weight size as the code embeddings. -HD
    def _codes_embeddings_init(self, codes_embeddings, dicts):
        codes_embeddingss = KeyedVectors.load_word2vec_format(codes_embeddings)
        # classmethod load_word2vec_format(fname, fvocab=None, binary=False,
        #   encoding='utf8', unicode_errors='strict',
        #   limit=None, datatype=<class 'numpy.float32'>)
        # Load the input-hidden weight matrix from the original
        #   C word2vec-tool format.
        weights = np.zeros(self.final.weight.size())
        for i in range(self.number_labels):
            code = dicts['ind2c'][i]
            weights[i] = codes_embeddingss[code]
        # set weight as the code embeddings.
        self.final.weight.data = torch.Tensor(weights).clone()

    def forward(self, x, target, desc_data=None, get_attention=False):
        # get embeddings
        x = self.embed(x)

        if self.pool == 'avg':
            # average pooling, works but horrible performance
            x = torch.mean(x, 1)
        elif self.pool == 'max':
            # TODO: read https://pytorch.org/docs/stable/generated/torch.nn.MaxPool1d.html#torch.nn.MaxPool1d  # noqa
            # which size dim to take?
            # torch.Size([16, 212, 100]), embed dim is 100
            # mat1 and mat2 shapes cannot be multiplied (3392x1 and 100x50)
            logging.info("kernel size used: {}".format(x.size()))
            # kernel_size=x.size()[2]
            x = F.max_pool1d(input=x, kernel_size=1)
        else:
            # average pooling
            # TODO: log error???
            x = torch.mean(x, 1)

        logits = torch.sigmoid(input=self.final(x))
        # only using the pooled, document embedding for logistic regression.
        # In this case, it is also possible to apply SVM for the task.
        loss = self._get_loss(logits, target, diffs=desc_data)
        return logits, loss, None


class BasicCNN(BaseModel):

    def __init__(self, number_labels, embeddings_file, kernel_size,
                 filter_maps, gpu=True, dicts=None, embedding_size=100,
                 dropout=0.5):
        super(BasicCNN, self).__init__(
            number_labels, embeddings_file, dicts,
            dropout=dropout, embedding_size=embedding_size)
        # initialize conv layer as in 2.1
        self.conv = nn.Conv1d(
            in_channels=self.embedding_size, out_channels=filter_maps,
            kernel_size=kernel_size)
        xavier_uniform(tensor=self.conv.weight)

        # linear output
        self.fc = nn.Linear(
            in_features=filter_maps, out_features=number_labels)
        xavier_uniform(tensor=self.fc.weight)
        logging.info("Done initializing vanilla CNN")

    def forward(self, x, target, desc_data=None, get_attention=False):
        # embed
        x = self.embed(x)
        x = self.embed_drop(x)
        x = x.transpose(1, 2)

        # conv/max-pooling
        c = self.conv(x)
        if get_attention:
            # get argmax vector too
            x, argmax = F.max_pool1d(
                torch.tanh(c), kernel_size=c.size()[2], return_indices=True)
            attn = self.construct_attention(argmax, c.size()[2])
        else:
            x = F.max_pool1d(torch.tanh(c), kernel_size=c.size()[2])
            attn = None
        x = x.squeeze(dim=2)

        # linear output
        x = self.fc(x)

        # final sigmoid to get predictions
        yhat = x
        loss = self._get_loss(yhat, target)
        return yhat, loss, attn

    def construct_attention(self, argmax, num_windows):
        attn_batches = []
        for argmax_i in argmax:
            attns = []
            for i in range(num_windows):
                # generate mask to select indices of conv features
                # where max was i
                mask = (argmax_i == i).repeat(1, self.number_labels).t()
                # apply mask to every label's weight vector and take the sum
                # to get the 'attention' score
                weights = self.fc.weight[mask].view(-1, self.number_labels)
                if len(weights.size()) > 0:
                    window_attns = weights.sum(dim=0)
                    attns.append(window_attns)
                else:
                    # this window was never a max
                    attns.append(
                        Variable(torch.zeros(self.number_labels)).cuda())
            # combine
            attn = torch.stack(attns)
            attn_batches.append(attn)
        attn_full = torch.stack(attn_batches)
        # put it in the right form for passing to interpret
        attn_full = attn_full.transpose(1, 2)
        return attn_full


class RNN(BaseModel):
    """
    input_size: The number of expected features in the input `x`
    hidden_size: The number of features in the hidden state `h`
    num_layers: Number of recurrent layers. E.g., setting ``num_layers=2``
        would mean stacking two GRUs together to form a `stacked GRU`,
        with the second GRU taking in outputs of the first GRU and
        computing the final results. Default: 1
    bias: If ``False``, then the layer does not use bias weights `b_ih` and
        `b_hh`. Default: ``True``
    batch_first: If ``True``, then the input and output tensors are provided
        as `(batch, seq, feature)` instead of `(seq, batch, feature)`.
        Note that this does not apply to hidden or cell states. See the
        Inputs/Outputs sections below for details.  Default: ``False``
    dropout: If non-zero, introduces a `Dropout` layer on the outputs of each
        GRU layer except the last layer, with dropout probability equal to
        :attr:`dropout`. Default: 0
    bidirectional: If ``True``, becomes a bidirectional GRU. Default: ``False``
    """
    def __init__(self, number_labels, embeddings_file, dicts, rnn_dim,
                 rnn_cell_type, rnn_layers, dropout, gpu, batch_size,
                 embedding_size, bidirectional):

        super(RNN, self).__init__(
            number_labels, embeddings_file, dicts, dropout, gpu=gpu,
            embedding_size=embedding_size)
        self.rnn_dim = rnn_dim
        self.rnn_cell_type = rnn_cell_type
        self.rnn_layers = rnn_layers
        self.bidirectional = bidirectional
        self.directions = 2 if bidirectional else 1
        self.batch_size = batch_size
        self.dropout = dropout

        # define recurrent cell
        if self.rnn_cell_type == 'gru':

            self.rnn = nn.GRU(
                input_size=self.embedding_size,
                hidden_size=floor(self.rnn_dim / self.directions),
                num_layers=self.rnn_layers,
                bias=True,
                batch_first=False,
                dropout=self.dropout,
                bidirectional=self.bidirectional)
        elif self.rnn_cell_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size=self.self.embedding_size,
                hidden_size=floor(self.rnn_dim / self.directions),
                num_layers=self.rnn_layers,
                batch_first=False,
                dropout=self.dropout,
                bidirectional=self.bidirectional,
                projection_size=0)
        else:
            logging.error("Invalid RNN cell type: {}".format(
                self.rnn_cell_type))

        # final linear output layer
        self.final = nn.Linear(
            in_features=self.rnn_dim,
            out_features=self.number_labels,
            bias=True)

        # arbitrary initialization
        # depends on cell type: GRU vs LSTM,
        # and GPU
        self.hidden = self.initialize_hidden_layer()

    def forward(self, x, target, desc_data=None, get_attention=False):

        # reset batch size at the start of each batch, clear hidden state
        self.reinitialize(batch_size=x.size()[0])

        # embed x, [251, 16, 100]
        embeds = self.embed(x).transpose(0, 1)
        # logging.debug(
        #     'shape of embeds: {}. Expect either (N, L, H_in), if '
        #     'batch_first=True, or (L, N, H_in)'.format(embeds.shape))

        # apply RNN, output: (N, L, D * H_out) if batch_first=True,
        # else, swap N and L
        # input=embeds, h_0=self.hidden
        output, self.hidden = self.rnn(embeds, self.hidden)
        # [251, 16, 100]
        # logging.debug(
        #     'shape of outputs: {}. Expect either (N, L, D * H_out), if '
        #     'batch_first=True, or (L, N, D* H_out)'.format(output.shape))

        # get final hidden state in the appropriate way
        # GRU's hidden: h_0
        # LSTM's hidden: (h_0, c_0)
        if self.rnn_cell_type == 'gru':
            last_hidden = self.hidden
        else:
            last_hidden = self.hidden[0]

        if self.directions == 1:
            last_hidden = last_hidden[-1]
        else:
            last_hidden = last_hidden[-2:].transpose(0, 1).contiguous().view(
                self.batch_size, -1)

        # apply linear layer and sigmoid to get predictions
        yhat = self.final(last_hidden)
        loss = self._get_loss(yhat, target)
        return yhat, loss, None

    def initialize_hidden_layer(self):
        if self.gpu:
            h_0 = Variable(
                torch.cuda.FloatTensor(
                    self.directions * self.rnn_layers,
                    self.batch_size,
                    floor(self.rnn_dim / self.directions)
                ).zero_()
            )

            if self.rnn_cell_type == 'gru':
                return h_0
            else:
                c_0 = Variable(
                    torch.cuda.FloatTensor(
                        self.directions * self.rnn_layers,
                        self.batch_size,
                        floor(self.rnn_dim / self.directions)
                    ).zero_()
                )
                return (h_0, c_0)
        else:
            h_0 = Variable(
                torch.zeros(
                    self.directions * self.rnn_layers,
                    self.batch_size,
                    floor(self.rnn_dim / self.directions)
                )
            )
            if self.rnn_cell_type == 'gru':
                return h_0
            else:
                c_0 = Variable(
                    torch.zeros(
                        self.directions * self.rnn_layers,
                        self.batch_size,
                        floor(self.rnn_dim / self.directions)
                    )
                )
                return (h_0, c_0)

    def reinitialize(self, batch_size):
        self.batch_size = batch_size
        self.hidden = self.initialize_hidden_layer()
