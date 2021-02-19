import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, tagset, embedding_dim, hidden_dim,
                 num_layers, bidirectional, dropout, pretrained=None):
        super(BiLSTM, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.tagset_size = len(tagset)
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        if pretrained is not None:
            self.word_embeds = nn.Embedding.from_pretrained(pretrained)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim // 2 if bidirectional else hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)
        self.hidden = None

    def init_hidden(self, batch_size, device):
        init_hidden_dim = self.hidden_dim // 2 if self.bidirectional else self.hidden_dim
        init_first_dim = self.num_layers * 2 if self.bidirectional else self.num_layers
        self.hidden = (
            torch.randn(init_first_dim, batch_size, init_hidden_dim).to(device),
            torch.randn(init_first_dim, batch_size, init_hidden_dim).to(device)
        )

    def repackage_hidden(self, hidden):
        """Wraps hidden states in new Tensors, to detach them from their history."""
        if isinstance(hidden, torch.Tensor):
            return hidden.detach_()
        else:
            return tuple(self.repackage_hidden(h) for h in hidden)

    def forward(self, batch_input, batch_input_lens, batch_mask):
        batch_size, padding_length = batch_input.size()
        batch_input = self.word_embeds(batch_input)  # size: #batch * padding_length * embedding_dim
        batch_input = rnn_utils.pack_padded_sequence(
            batch_input, batch_input_lens, batch_first=True)
        batch_output, self.hidden = self.lstm(batch_input, self.hidden)
        self.repackage_hidden(self.hidden)
        batch_output, _ = rnn_utils.pad_packed_sequence(batch_output, batch_first=True)
        batch_output = batch_output.contiguous().view(batch_size * padding_length, -1)
        batch_output = batch_output[batch_mask, ...]
        out = self.hidden2tag(batch_output)
        return out

    def neg_log_likelihood(self, batch_input, batch_input_lens, batch_mask, batch_target):
        loss = nn.CrossEntropyLoss(reduction='mean')
        feats = self(batch_input, batch_input_lens, batch_mask)
        batch_target = torch.cat(batch_target, 0)
        return loss(feats, batch_target)

    def predict(self, batch_input, batch_input_lens, batch_mask):
        feats = self(batch_input, batch_input_lens, batch_mask)
        val, pred = torch.max(feats, 1)
        return pred


class CRF(nn.Module):
    def __init__(self, tagset, start_tag, end_tag, device):
        super(CRF, self).__init__()
        self.tagset_size = len(tagset)
        self.START_TAG_IDX = tagset.index(start_tag)
        self.END_TAG_IDX = tagset.index(end_tag)
        self.START_TAG_TENSOR = torch.LongTensor([self.START_TAG_IDX], device=device)
        self.END_TAG_TENSOR = torch.LongTensor([self.END_TAG_IDX], device=device)
        # trans: (tagset_size, tagset_size) trans (i, j) means state_i -> state_j
        self.trans = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size)
        )
        # self.trans.data[...] = 1
        self.trans.data[:, self.START_TAG_IDX] = -10000
        self.trans.data[self.END_TAG_IDX, :] = -10000
        self.device = device

    def init_alpha(self, batch_size, tagset_size):
        return torch.full((batch_size, tagset_size, 1), -10000, dtype=torch.float, device=self.device)

    def init_path(self, size_shape):
        # Initialization Path - LongTensor + Device + Full_value=0
        return torch.full(size_shape, 0, dtype=torch.long, device=self.device)

    def _iter_legal_batch(self, batch_input_lens, reverse=False):
        index = torch.range(0, batch_input_lens.sum() - 1, dtype=torch.long)
        packed_index = rnn_utils.pack_sequence(
            torch.split(index, batch_input_lens.tolist())
        )
        batch_iter = torch.split(packed_index.data, packed_index.batch_sizes.tolist())
        batch_iter = reversed(batch_iter) if reverse else batch_iter
        for idx in batch_iter:
            yield idx, idx.size()[0]

    def score_z(self, feats, batch_input_lens):
        # 模拟packed pad过程
        tagset_size = feats.shape[1]
        batch_size = len(batch_input_lens)
        alpha = self.init_alpha(batch_size, tagset_size)
        alpha[:, self.START_TAG_IDX, :] = 0  # Initialization
        for legal_idx, legal_batch_size in self._iter_legal_batch(batch_input_lens):
            feat = feats[legal_idx, ].view(legal_batch_size, 1, tagset_size)  # 
            # #batch * 1 * |tag| + #batch * |tag| * 1 + |tag| * |tag| = #batch * |tag| * |tag|
            legal_batch_score = feat + alpha[:legal_batch_size, ] + self.trans
            alpha_new = torch.logsumexp(legal_batch_score, 1).unsqueeze(2)
            alpha[:legal_batch_size, ] = alpha_new
        alpha = alpha + self.trans[:, self.END_TAG_IDX].unsqueeze(1)
        score = torch.logsumexp(alpha, 1).sum()
        return score

    def score_sentence(self, feats, batch_target):
        # CRF Batched Sentence Score
        # feats: (#batch_state(#words), tagset_size)
        # batch_target: list<torch.LongTensor> At least One LongTensor
        # Warning: words order =  batch_target order
        def _add_start_tag(target):
            return torch.cat([self.START_TAG_TENSOR, target])

        def _add_end_tag(target):
            return torch.cat([target, self.END_TAG_TENSOR])

        from_state = [_add_start_tag(target) for target in batch_target]
        to_state = [_add_end_tag(target) for target in batch_target]
        from_state = torch.cat(from_state)  
        to_state = torch.cat(to_state)  
        trans_score = self.trans[from_state, to_state]

        gather_target = torch.cat(batch_target).view(-1, 1)
        emit_score = torch.gather(feats, 1, gather_target)  

        return trans_score.sum() + emit_score.sum()

    def viterbi(self, feats, batch_input_lens):
        word_size, tagset_size = feats.shape
        batch_size = len(batch_input_lens)
        viterbi_path = self.init_path(feats.shape)  # use feats.shape to init path.shape
        alpha = self.init_alpha(batch_size, tagset_size)
        alpha[:, self.START_TAG_IDX, :] = 0  # Initialization
        for legal_idx, legal_batch_size in self._iter_legal_batch(batch_input_lens):
            feat = feats[legal_idx, :].view(legal_batch_size, 1, tagset_size)
            legal_batch_score = feat + alpha[:legal_batch_size, ] + self.trans
            alpha_new, best_tag = torch.max(legal_batch_score, 1)
            alpha[:legal_batch_size, ] = alpha_new.unsqueeze(2)
            viterbi_path[legal_idx, ] = best_tag
        alpha = alpha + self.trans[:, self.END_TAG_IDX].unsqueeze(1)
        path_score, best_tag = torch.max(alpha, 1)
        path_score = path_score.squeeze()  # path_score=#batch

        best_paths = self.init_path((word_size, 1))
        for legal_idx, legal_batch_size in self._iter_legal_batch(batch_input_lens, reverse=True):
            best_paths[legal_idx, ] = best_tag[:legal_batch_size, ]  # 
            backword_path = viterbi_path[legal_idx, ]  # 1 * |Tag|
            this_tag = best_tag[:legal_batch_size, ]  # 1 * |legal_batch_size|
            backword_tag = torch.gather(backword_path, 1, this_tag)
            best_tag[:legal_batch_size, ] = backword_tag
            # never computing <START>

        # best_paths = #words
        return path_score.view(-1), best_paths.view(-1)


class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tagset, embedding_dim, hidden_dim,
                 num_layers, bidirectional, dropout, start_tag, end_tag, device, pretrained=None):
        super(BiLSTM_CRF, self).__init__()
        self.bilstm = BiLSTM(vocab_size, tagset, embedding_dim, hidden_dim,
                             num_layers, bidirectional, dropout, pretrained)
        self.CRF = CRF(tagset, start_tag, end_tag, device)

    def init_hidden(self, batch_size, device):
        self.bilstm.hidden = self.bilstm.init_hidden(batch_size, device)

    def forward(self, batch_input, batch_input_lens, batch_mask):
        feats = self.bilstm(batch_input, batch_input_lens, batch_mask)
        score, path = self.CRF.viterbi(feats, batch_input_lens)
        return path

    def neg_log_likelihood(self, batch_input, batch_input_lens, batch_mask, batch_target):
        feats = self.bilstm(batch_input, batch_input_lens, batch_mask)
        gold_score = self.CRF.score_sentence(feats, batch_target)
        forward_score = self.CRF.score_z(feats, batch_input_lens)
        return forward_score - gold_score

    def predict(self, batch_input, batch_input_lens, batch_mask):
        return self(batch_input, batch_input_lens, batch_mask)