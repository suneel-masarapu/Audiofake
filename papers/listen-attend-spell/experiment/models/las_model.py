import torch
if torch.cuda.is_available():
    import torch.cuda as device
else:
    import torch as device
from torch.autograd import Variable    
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from utils.functions import TimeDistributed, CreateOnehotVariable
import numpy as np


# BLSTM layer for pBLSTM
class pBLSTMLayer(nn.Module):
    def __init__(self, input_feature_dim, hidden_dim, rnn_unit='LSTM', dropout_rate=0.0):
        super(pBLSTMLayer, self).__init__()
        self.rnn_unit = getattr(nn, rnn_unit.upper())
        
        # feature dimension will be doubled since time resolution reduction
        self.BLSTM = self.rnn_unit(input_feature_dim * 2, hidden_dim, 1, 
                                   bidirectional=True, dropout=dropout_rate, batch_first=True)
    
    def forward(self, input_x):
        batch_size = input_x.size(0)
        timestep = input_x.size(1)
        feature_dim = input_x.size(2)
        
        # Reduce time resolution
        if timestep % 2 != 0:
            input_x = input_x[:, :-1, :]  # Drop the last frame to make it even
        
        input_x = input_x.contiguous().view(batch_size, int(input_x.size(1)/2), feature_dim * 2)
        
        # Bidirectional RNN
        output, hidden = self.BLSTM(input_x)
        return output, hidden


# Listener is a pBLSTM stacking layers to reduce time resolution
class Listener(nn.Module):
    def __init__(self, input_feature_dim, listener_hidden_dim, listener_layer, rnn_unit, use_gpu, dropout_rate=0.0, **kwargs):
        super(Listener, self).__init__()
        self.listener_layer = listener_layer
        assert self.listener_layer >= 1, 'Listener should have at least 1 layer'
        
        self.pLSTM_layer0 = pBLSTMLayer(input_feature_dim, listener_hidden_dim, 
                                        rnn_unit=rnn_unit, dropout_rate=dropout_rate)
        
        for i in range(1, self.listener_layer):
            setattr(self, 'pLSTM_layer' + str(i), 
                    pBLSTMLayer(listener_hidden_dim * 2, listener_hidden_dim, 
                                rnn_unit=rnn_unit, dropout_rate=dropout_rate))
        
        self.use_gpu = use_gpu
        if self.use_gpu:
            self = self.cuda()
    
    def forward(self, input_x):
        output, _ = self.pLSTM_layer0(input_x)
        for i in range(1, self.listener_layer):
            output, _ = getattr(self, 'pLSTM_layer' + str(i))(output)
        
        return output


# Speller specified in the paper
class Speller(nn.Module):
    def __init__(self, output_class_dim, speller_hidden_dim, rnn_unit, speller_rnn_layer, use_gpu, max_label_len,
                 use_mlp_in_attention, mlp_dim_in_attention, mlp_activate_in_attention, listener_hidden_dim,
                 multi_head, decode_mode, **kwargs):
        super(Speller, self).__init__()
        self.rnn_unit = getattr(nn, rnn_unit.upper())
        self.max_label_len = max_label_len
        self.decode_mode = decode_mode
        self.use_gpu = use_gpu
        self.float_type = torch.cuda.FloatTensor if use_gpu else torch.FloatTensor
        self.label_dim = output_class_dim
        self.speller_hidden_dim = speller_hidden_dim
        
        # RNN layer - input is (output_class_dim + context_dim)
        self.rnn_layer = self.rnn_unit(output_class_dim + speller_hidden_dim, speller_hidden_dim,
                                       num_layers=speller_rnn_layer, batch_first=True)
        
        # Attention mechanism
        self.attention = Attention(
            mlp_preprocess_input=use_mlp_in_attention,
            preprocess_mlp_dim=mlp_dim_in_attention,
            activate=mlp_activate_in_attention,
            input_feature_dim=speller_hidden_dim,
            listener_feature_dim=2 * listener_hidden_dim,  # pBLSTM is bidirectional
            multi_head=multi_head
        )
        
        # Output projection
        self.character_distribution = nn.Linear(speller_hidden_dim + 2 * listener_hidden_dim, output_class_dim)
        self.softmax = nn.LogSoftmax(dim=-1)
        
        # Project listener features to speller hidden dimension for initial context
        self.context_projection = nn.Linear(2 * listener_hidden_dim, speller_hidden_dim)
        
        if self.use_gpu:
            self = self.cuda()
    
    def forward_step(self, input_word, last_hidden_state, listener_feature):
        rnn_output, hidden_state = self.rnn_layer(input_word, last_hidden_state)
        attention_score, context = self.attention(rnn_output, listener_feature)
        
        # Concatenate RNN output with context for final prediction
        concat_feature = torch.cat([rnn_output.squeeze(dim=1), context], dim=-1)
        raw_pred = self.softmax(self.character_distribution(concat_feature))
        
        return raw_pred, hidden_state, context, attention_score
    
    def forward(self, listener_feature, ground_truth=None, teacher_force_rate=0.9):
        if ground_truth is None:
            teacher_force_rate = 0
        teacher_force = np.random.random_sample() < teacher_force_rate
        
        batch_size = listener_feature.size(0)
        
        # Initialize with start token (usually zeros or a special start token)
        output_word = CreateOnehotVariable(self.float_type(np.zeros((batch_size, 1))), self.label_dim)
        if self.use_gpu:
            output_word = output_word.cuda()
        
        # Initial context from first timestep of listener features
        init_context = listener_feature[:, 0, :]  # shape: [batch, 2*listener_hidden_dim]
        projected_context = self.context_projection(init_context)
        rnn_input = torch.cat([output_word, projected_context.unsqueeze(1)], dim=-1)
        
        hidden_state = None
        raw_pred_seq = []
        attention_record = []
        
        max_step = self.max_label_len if ground_truth is None or not teacher_force else ground_truth.size(1)
        
        for step in range(max_step):
            raw_pred, hidden_state, context, attention_score = self.forward_step(rnn_input, hidden_state, listener_feature)
            raw_pred_seq.append(raw_pred)
            attention_record.append(attention_score)
            
            # Prepare next input
            if teacher_force and ground_truth is not None:
                if step + 1 < ground_truth.size(1):
                    output_word = ground_truth[:, step, :].unsqueeze(1).type(self.float_type)
                else:
                    # Use predicted output for the last step
                    if self.decode_mode == 0:
                        output_word = raw_pred.unsqueeze(1)
                    elif self.decode_mode == 1:
                        output_word = torch.zeros_like(raw_pred)
                        for idx, i in enumerate(raw_pred.topk(1)[1]):
                            output_word[idx, int(i)] = 1
                        output_word = output_word.unsqueeze(1)
                    else:
                        sampled_word = Categorical(F.softmax(raw_pred, dim=-1)).sample()
                        output_word = torch.zeros_like(raw_pred)
                        for idx, i in enumerate(sampled_word):
                            output_word[idx, int(i)] = 1
                        output_word = output_word.unsqueeze(1)
            else:
                # Inference mode
                if self.decode_mode == 0:
                    output_word = F.softmax(raw_pred, dim=-1).unsqueeze(1)
                elif self.decode_mode == 1:
                    output_word = torch.zeros_like(raw_pred)
                    for idx, i in enumerate(raw_pred.topk(1)[1]):
                        output_word[idx, int(i)] = 1
                    output_word = output_word.unsqueeze(1)
                else:
                    sampled_word = Categorical(F.softmax(raw_pred, dim=-1)).sample()
                    output_word = torch.zeros_like(raw_pred)
                    for idx, i in enumerate(sampled_word):
                        output_word[idx, int(i)] = 1
                    output_word = output_word.unsqueeze(1)
            
            # Project context for next RNN input
            projected_context = self.context_projection(context)
            rnn_input = torch.cat([output_word, projected_context.unsqueeze(1)], dim=-1)
        
        return raw_pred_seq, attention_record


# Attention mechanism
class Attention(nn.Module):
    def __init__(self, mlp_preprocess_input, preprocess_mlp_dim, activate, mode='dot', 
                 input_feature_dim=512, listener_feature_dim=512, multi_head=1):
        super(Attention, self).__init__()
        self.mode = mode.lower()
        self.mlp_preprocess_input = mlp_preprocess_input
        self.multi_head = multi_head
        self.softmax = nn.Softmax(dim=-1)
        self.listener_feature_dim = listener_feature_dim
        
        if mlp_preprocess_input:
            self.preprocess_mlp_dim = preprocess_mlp_dim
            # Project decoder state
            self.phi = nn.Linear(input_feature_dim, preprocess_mlp_dim * multi_head)
            # Project listener features  
            self.psi = nn.Linear(listener_feature_dim, preprocess_mlp_dim)
            
            if self.multi_head > 1:
                self.dim_reduce = nn.Linear(listener_feature_dim * multi_head, listener_feature_dim)
            
            if activate != 'None':
                self.activate = getattr(F, activate)
            else:
                self.activate = None
    
    def forward(self, decoder_state, listener_feature):
        if self.mlp_preprocess_input:
            if self.activate:
                comp_decoder_state = self.activate(self.phi(decoder_state))
                comp_listener_feature = self.activate(TimeDistributed(self.psi, listener_feature))
            else:
                comp_decoder_state = self.phi(decoder_state)
                comp_listener_feature = TimeDistributed(self.psi, listener_feature)
        else:
            comp_decoder_state = decoder_state
            comp_listener_feature = listener_feature
        
        if self.mode == 'dot':
            if self.multi_head == 1:
                energy = torch.bmm(comp_decoder_state, comp_listener_feature.transpose(1, 2)).squeeze(dim=1)
                attention_score = [self.softmax(energy)]
                context = torch.sum(listener_feature * attention_score[0].unsqueeze(2).repeat(1, 1, listener_feature.size(2)), dim=1)
            else:
                attention_score = [self.softmax(torch.bmm(att_query, comp_listener_feature.transpose(1, 2)).squeeze(dim=1))
                                   for att_query in torch.split(comp_decoder_state, self.preprocess_mlp_dim, dim=-1)]
                projected_src = [torch.sum(listener_feature * att_s.unsqueeze(2).repeat(1, 1, listener_feature.size(2)), dim=1)
                                 for att_s in attention_score]
                context = self.dim_reduce(torch.cat(projected_src, dim=-1))
        else:
            # TODO: other attention implementations
            pass
        
        return attention_score, context