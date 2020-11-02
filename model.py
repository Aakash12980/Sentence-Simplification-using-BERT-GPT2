from transformers import EncoderDecoderModel, BertConfig
from torch.nn import Module
import torch.nn as nn
from tokenizer import Tokenizer
import torch
import shutil
import os

class EncDecModel(Module):
    def __init__(self, max_len=80):
        super(EncDecModel, self).__init__()
        self.tokenizer = Tokenizer(max_len)        
        self.model = EncoderDecoderModel.from_encoder_decoder_pretrained('bert-base-cased', 'gpt2')
        self.model.config.decoder_start_token_id = self.tokenizer.gpt2_tokenizer.bos_token_id
        self.model.config.eos_token_id = self.tokenizer.gpt2_tokenizer.eos_token_id
        self.model.config.no_repeat_ngram_size = 3
        # self.model.early_stopping = True
        # self.model.length_penalty = 2.0
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, batch, device, train=True):
        src_tensors, src_attn_tensors, tgt_tensors, tgt_attn_tensors, labels = self.tokenizer.encode_batch(batch)
        loss, logits = self.model(input_ids = src_tensors.to(device), 
                            decoder_input_ids = tgt_tensors.to(device),
                            attention_mask = src_attn_tensors.to(device),
                            decoder_attention_mask = tgt_attn_tensors.to(device),
                            labels = labels.to(device))[:2]
        if not train:
            outputs = self.softmax(logits)
            return loss, torch.argmax(outputs, dim=-1)
        else:
            return loss

    def load_model(self, checkpt_path, device, optimizer=None):

        if device == "cpu":
            checkpoint = torch.load(checkpt_path, map_location=torch.device("cpu"))
            self.load_state_dict(checkpoint["model_state_dict"])
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"], map_location=torch.device("cpu"))
            
        else:
            checkpoint = torch.load(checkpt_path)
            self.load_state_dict(checkpoint["model_state_dict"])
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        eval_loss = checkpoint["eval_loss"]
        epoch = checkpoint["epoch"]

        return optimizer, eval_loss, epoch
    
    def save_checkpt(self, state, is_best, check_pt_path, best_model_path):
        f_path = check_pt_path
        torch.save(state, f_path)

        if is_best:
            best_fpath = best_model_path
            shutil.copyfile(f_path, best_fpath)

