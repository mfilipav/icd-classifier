from gensim.models import KeyedVectors
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
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
        y = self.fc(x)

        # final sigmoid to get predictions
        yhat = y
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


class CAML(BaseModel):
    """
    Copied and re-formatted from Mullenbach 2018
    """

    def __init__(self, number_labels, embeddings_file, kernel_size,
                 filter_maps, lmbda, gpu, dicts, embedding_size=100,
                 dropout=0.5, code_emb=None):
        super(CAML, self).__init__(
            number_labels, embeddings_file, dicts, lmbda, dropout=dropout,
            gpu=gpu, embedding_size=embedding_size)

        # initialize conv layer as in 2.1
        self.conv = nn.Conv1d(
            in_channels=self.embedding_size, out_channels=filter_maps,
            kernel_size=kernel_size, padding=int(floor(kernel_size/2)))
        xavier_uniform(self.conv.weight)

        # per-label context vectors for computing attention as in 2.2
        self.U = nn.Linear(
            in_features=filter_maps, out_features=number_labels)
        xavier_uniform(tensor=self.U.weight)

        # final layer: create a matrix to use for the L binary
        # classifiers as in 2.3
        self.final = nn.Linear(
            in_features=filter_maps, out_features=number_labels)
        xavier_uniform(self.final.weight)

        # initialize with trained code embeddings if applicable, DR-CAML
        if code_emb:
            self._code_emb_init(code_emb, dicts)
            # also set conv weights to do sum of inputs
            # expand 3rd dim to be replicated of 'kernel_size'
            # (first two dims leave untouched -1, -1)
            weights = torch.eye(
                n=self.embedding_size).unsqueeze(dim=2).expand(
                    -1, -1, kernel_size) / kernel_size
            self.conv.weight.data = weights.clone()
            self.conv.bias.data.zero_()

        # besides 'self.embed', created at BaseModel.__init__, create
        # description embeddings.
        # Convolution applied to label descriptions as in 2.5
        # description module has its own embedding and convolution layers
        if lmbda > 0:
            logging.info("Initialize description embeddings")
            W = self.embed.weight.data
            num_embeddings = W.size()[0]
            embedding_dim = W.size()[1]
            self.desc_embedding = nn.Embedding(
                num_embeddings=num_embeddings, embedding_dim=embedding_dim,
                padding_idx=0)
            self.desc_embedding.weight.data = W.clone()
            logging.info(
                "size of description embeddings: {}, "
                "size of each desc embedding vector: {}".format(
                        num_embeddings, embedding_dim))

            self.label_conv = nn.Conv1d(
                in_channels=self.embedding_size, out_channels=filter_maps,
                kernel_size=kernel_size, padding=int(floor(kernel_size/2)))
            xavier_uniform(self.label_conv.weight)

            self.label_fc1 = nn.Linear(
                in_features=filter_maps, out_features=filter_maps)
            xavier_uniform(self.label_fc1.weight)

    def _code_emb_init(self, code_emb, dicts):
        code_embeddings = KeyedVectors.load_word2vec_format(fname=code_emb)
        weights = np.zeros(shape=self.final.weight.size())
        for i in range(self.number_labels):
            code = dicts['ind2c'][i]
            weights[i] = code_embeddings[code]
        self.U.weight.data = torch.Tensor(weights).clone()
        self.final.weight.data = torch.Tensor(weights).clone()

    def forward(self, x, target, desc_data=None, get_attention=True):
        # get embeddings and apply dropout
        x = self.embed(x)
        x = self.embed_drop(x)
        x = x.transpose(1, 2)

        # apply convolution and nonlinearity (tanh)
        x = torch.tanh(self.conv(x).transpose(1, 2))
        # apply attention
        alpha = F.softmax(
            self.U.weight.matmul(x.transpose(1, 2)), dim=2)

        # document representations are weighted sums using the attention.
        # Can compute all at once as a matmul
        m = alpha.matmul(x)

        # final layer classification
        y = self.final.weight.mul(m).sum(dim=2).add(self.final.bias)

        if desc_data is not None:
            # run descriptions through description module
            b_batch = self.embed_descriptions(desc_data, self.gpu)
            # get l2 similarity loss
            diffs = self._compare_label_embeddings(target, b_batch, desc_data)
        else:
            diffs = None

        # final sigmoid to get predictions
        yhat = y
        loss = self._get_loss(yhat, target, diffs)
        return yhat, loss, alpha


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
