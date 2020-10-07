
import click
import pickle
import os
import torch
import numpy as np
from utils import open_file
from tokenizer import create_sent_tokens, generate_tokens, get_sent_from_tokens
from dataset import WikiDataset
from torch.utils.data import DataLoader
from transformers import EncoderDecoderModel, BertConfig, EncoderDecoderConfig
import time
import shutil
# from torchtext.data.metrics import bleu_score
import tqdm
import logging
import gc

CONTEXT_SETTINGS = dict(help_option_names = ['-h', '--help'])

TRAIN_BATCH_SIZE = 4
N_EPOCH = 10
LOG_EVERY = 5000

config_encoder = BertConfig()
config_decoder = BertConfig()
# config_decoder = GPT2Config()
config_decoder.is_decoder = True
config_decoder.add_cross_attention = True
config = EncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)
model = EncoderDecoderModel.from_encoder_decoder_pretrained('bert-base-cased', 'bert-base-cased', config=config)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} as device")


def collate_fn(batch):
    data_list, label_list = [], []
    for _data, _label in batch:
        data_list.append(_data)
        label_list.append(_label)
    return data_list, label_list

def save_model_checkpt(state, is_best, check_pt_path, best_model_path):
    f_path = check_pt_path
    torch.save(state, f_path)

    if is_best:
        best_fpath = best_model_path
        shutil.copyfile(f_path, best_fpath)

def load_checkpt(checkpt_path, optimizer=None):
    checkpoint = torch.load(checkpt_path)
    if device == "cpu":
        model.load_state_dict(checkpoint["model_state_dict"], map_location=torch.device("cpu"))
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"], map_location=torch.device("cpu"))
    else:
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    eval_loss = checkpoint["eval_loss"]
    epoch = checkpoint["epoch"]

    return optimizer, eval_loss, epoch



def evaluate(batch_iter, e_loss):
    was_training = model.training
    model.eval()
    eval_loss = e_loss

    with torch.no_grad():
        for step, batch in enumerate(batch_iter):
            src_tensors, src_attn_tensors, tgt_tensors, tgt_attn_tensors = generate_tokens(batch)
            loss, _ = model(input_ids = src_tensors.to(device), 
                            decoder_input_ids = tgt_tensors.to(device),
                            attention_mask = src_attn_tensors.to(device),
                            decoder_attention_mask = tgt_attn_tensors.to(device),
                            labels=tgt_tensors.to(device))[:2]
            
            eval_loss += (1/(step+1)) * (loss.item() - eval_loss)

    if was_training:
        model.train()

    return eval_loss 

@click.group(context_settings=CONTEXT_SETTINGS)
@click.version_option(version = '1.0.0')
def task():
    ''' This is the documentation of the main file. This is the reference for executing this file.'''
    pass


@task.command()
@click.option('--src_train', default="./drive/My Drive/Mini Project/dataset/src_train.txt", help="train source file path")
@click.option('--tgt_train', default="./drive/My Drive/Mini Project/dataset/tgt_train.txt", help="train target file path")
@click.option('--src_valid', default="./drive/My Drive/Mini Project/dataset/src_valid.txt", help="validation source file path")
@click.option('--tgt_valid', default="./drive/My Drive/Mini Project/dataset/tgt_valid.txt", help="validation target file path")
@click.option('--best_model', default="./drive/My Drive/Mini Project/best_model/model.pt", help="best model file path")
@click.option('--checkpoint_path', default="./drive/My Drive/Mini Project/checkpoint/model_ckpt.pt", help=" model check point files path")
@click.option('--seed', default=123, help="manual seed value (default=123)")
def train(**kwargs):
    print("Training data module executing...")
    seed = kwargs["seed"]
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed(seed)
    
    np.random.seed(seed)
    print("Loading dataset...")
    
    src_train = open_file(kwargs['src_train'])
    tgt_train = open_file(kwargs['tgt_train'])
    src_valid = open_file(kwargs['src_valid'])
    tgt_valid = open_file(kwargs['tgt_valid'])
    train_len = len(src_train)
    valid_len = len(src_valid)
    print("Dataset Loaded.")

    train_dataset = WikiDataset(src_train, tgt_train)
    valid_dataset = WikiDataset(src_valid, tgt_valid)
    del src_valid, src_train, tgt_train, tgt_valid

    print("Creating Dataloader...")
    train_dl = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, collate_fn=collate_fn, shuffle=True)
    valid_dl = DataLoader(valid_dataset, batch_size=TRAIN_BATCH_SIZE, collate_fn=collate_fn, shuffle=True)
    print("Dataloader created.")

    model.to(device)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        'weight_decay_rate': 0.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=3e-5)
    eval_loss = float('inf')
    start_epoch = 0
    if os.path.exists(kwargs["checkpoint_path"]):
        optimizer, eval_loss, start_epoch = load_checkpt(kwargs["checkpoint_path"], optimizer)
        print(f"Loading model from checkpoint with start epoch: {start_epoch} and loss: {eval_loss}")

    train_model(start_epoch, eval_loss, (train_dl, valid_dl), optimizer, kwargs["checkpoint_path"], kwargs["best_model"], (train_len, valid_len))
    print("Model Training Complete!")
    
    

@task.command()
@click.option('--src_test', default="./drive/My Drive/Mini Project/dataset/src_test.txt", help="test source file path")
@click.option('--tgt_test', default="./drive/My Drive/Mini Project/dataset/tgt_test.txt", help="test target file path")
@click.option('--best_model', default="./drive/My Drive/Mini Project/best_model/model.pt", help="best model file path")
def test(**kwargs):
    print("Testing Model module executing...")

    src_test = open_file(kwargs['src_test'])
    tgt_test = open_file(kwargs['tgt_test'])
    len_data = len(src_test)

    test_dataset = WikiDataset(src_test, tgt_test)

    test_dl = DataLoader(test_dataset, batch_size=TRAIN_BATCH_SIZE, collate_fn=collate_fn, shuffle=True)

    _,_,_ = load_checkpt(kwargs["best_model"])
    print("Model loaded...")
    model.to(device)
    model.eval()

    test_start_time = time.time()
    epoch_test_loss = evaluate(test_dl, 0)
    epoch_test_loss = epoch_test_loss/len_data
    print(f'avg. test loss: {epoch_test_loss:.5f} | time elapsed: {time.time() - test_start_time}')
    print("Test Complete!")

@task.command()
@click.option('--src_file', default="./drive/My Drive/Mini Project/dataset/src_file.txt", help="test source file path")
@click.option('--best_model', default="./drive/My Drive/Mini Project/best_model/model.pt", help="best model file path")
@click.option('--output', default="./drive/My Drive/Mini Project/outputs/decoded.txt", help="file path to save predictions")
def decode(**kwargs):
    print("Decoding sentences module executing...")
    src_test = open_file(kwargs['src_file'])
    print("Saved model loading...")
    _,_,_ = load_checkpt(kwargs["best_model"])
    print("Model loaded.")
    model.to(device)
    model.eval()
    inp_tokens = create_sent_tokens(src_test)
    predicted_list = []
    print("Decoding Sentences...")
    for tensor in inp_tokens:
        with torch.no_grad():
            predicted = model.generate(tensor.to(device), decoder_start_token_id=model.config.decoder.pad_token_id)
            predicted_list.append(predicted.squeeze())
    
    output = get_sent_from_tokens(predicted_list)
    with open(kwargs["output"], "w") as f:
        for sent in output:
            f.write(sent + "\n")
    print("Output file saved successfully.")

def train_model(start_epoch, eval_loss, loaders, optimizer, check_pt_path, best_model_path, len_data):
    best_eval_loss = eval_loss
    print("Model training started...")
    for epoch in range(start_epoch, N_EPOCH):
        print("Epoch one running...")
        epoch_start_time = time.time()
        epoch_train_loss = 0
        epoch_eval_loss = 0

        model.train()
        for step, batch in enumerate(loaders[0]):

            src_tensors, src_attn_tensors, tgt_tensors, tgt_attn_tensors = generate_tokens(batch)

            optimizer.zero_grad()
            model.zero_grad()
            loss = model(input_ids = src_tensors.to(device), 
                            decoder_input_ids = tgt_tensors.to(device),
                            attention_mask = src_attn_tensors.to(device),
                            decoder_attention_mask = tgt_attn_tensors.to(device),
                            labels = tgt_tensors.to(device))[0]
            
            loss.backward()
            optimizer.step()
            epoch_train_loss += (1/(step+1))*(loss.item() - epoch_train_loss)

            if (step+1) % LOG_EVERY == 0:
                print(f'Epoch: {epoch} | iter: {step+1} | avg. loss: {epoch_train_loss} | time elapsed: {time.time() - epoch_start_time}')
        

        epoch_train_loss = epoch_train_loss/len_data[0]
        print(f'Completed Epoch: {epoch} | avg. train loss: {epoch_train_loss:.5f} | Total Epoch time: {time.time() - epoch_start_time}')

        eval_start_time = time.time()
        epoch_eval_loss = evaluate(loaders[1], epoch_eval_loss)
        epoch_eval_loss = epoch_eval_loss/len_data[1]
        print(f'Completed Epoch: {epoch} | avg. eval loss: {epoch_eval_loss:.5f} | time elapsed: {time.time() - eval_start_time}')
        
        check_pt = {
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'eval_loss': epoch_eval_loss
        }
        check_pt_time = time.time()
        print("Saving Checkpoint.......")
        if epoch_eval_loss < best_eval_loss:
            print("New best model found")
            best_eval_loss = epoch_eval_loss
            save_model_checkpt(check_pt, True, check_pt_path, best_model_path)
        else:
            save_model_checkpt(check_pt, False, check_pt_path, best_model_path)  
        print(f"Checkpoint saved successfully with time: {time.time() - check_pt_time}") 
        gc.collect()
        torch.cuda.empty_cache()     


if __name__ == "__main__":
    task()
    
