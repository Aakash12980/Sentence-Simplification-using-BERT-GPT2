import click
import pickle
import os
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from data import WikiDataset
from tokenizer import Tokenizer
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from transformers import EncoderDecoderModel, BertConfig, EncoderDecoderConfig, GPT2Tokenizer
import time
import tqdm
import logging
import gc
import shutil
import sari

TRAIN_BATCH_SIZE = 4
N_EPOCH = 6
max_token_len = 80
LOG_EVERY = 10000

logging.basicConfig(filename="./drive/My Drive/Mini Project/log_file.log", level=logging.INFO, 
                format="%(asctime)s:%(levelname)s: %(message)s")
CONTEXT_SETTINGS = dict(help_option_names = ['-h', '--help'])

model = EncoderDecoderModel.from_encoder_decoder_pretrained('bert-base-cased', 'gpt2')
model.decoder.config.use_cache = False
tokenizer = Tokenizer(max_token_len)
model.config.decoder_start_token_id = tokenizer.gpt2_tokenizer.bos_token_id
model.config.eos_token_id = tokenizer.gpt2_tokenizer.eos_token_id
model.config.max_length = max_token_len
model.config.no_repeat_ngram_size = 3

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} as device")
model.to(device)

def collate_fn(batch):
    data_list, label_list, ref_list = [], [], []
    for _data, _label, _ref in batch:
        data_list.append(_data)
        label_list.append(_label)
        ref_list.append(_ref)
    return data_list, label_list, ref_list

def compute_bleu_score(logits, labels):
    refs = Tokenizer.get_sent_tokens(labels)
    weights = (1.0/2.0, 1.0/2.0, )
    score = corpus_bleu(refs, logits.tolist(), smoothing_function=SmoothingFunction(epsilon=1e-10).method1, weights=weights)
    return score

def compute_sari(norm, pred_tensor, ref):
    pred = tokenizer.decode_sent_tokens(pred_tensor)
    score = 0
    for step, item in enumerate(ref):
        score += sari.SARIsent(norm[step], pred[step], item)
    return score/TRAIN_BATCH_SIZE

def evaluate(data_loader, e_loss):
    was_training = model.training
    model.eval()
    eval_loss = e_loss
    bleu_score = 0
    sari_score = 0
    softmax = nn.LogSoftmax(dim = -1)

    with torch.no_grad():
        for step, batch in enumerate(data_loader):
            src_tensors, src_attn_tensors, tgt_tensors, tgt_attn_tensors, labels = tokenizer.encode_batch(batch)
            loss, logits = model(input_ids = src_tensors.to(device), 
                            decoder_input_ids = tgt_tensors.to(device),
                            attention_mask = src_attn_tensors.to(device),
                            decoder_attention_mask = tgt_attn_tensors.to(device),
                            labels = labels.to(device))[:2]
            outputs = softmax(logits)
            score = compute_bleu_score(torch.argmax(outputs, dim=-1), batch[1])
            s_score = compute_sari(batch[0], torch.argmax(outputs, dim=-1), batch[2])
            if step == 0:
                eval_loss = loss.item()
                bleu_score = score
                sari_score = s_score
            else:
                eval_loss = (1/2.0)*(eval_loss + loss.item())
                bleu_score = (1/2.0)* (bleu_score+score)
                sari_score = (1/2.0)* (bleu_score+s_score)
        
    if was_training:
        model.train()

    return eval_loss, bleu_score, sari_score

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

def save_model_checkpt(state, is_best, check_pt_path, best_model_path):
    f_path = check_pt_path
    torch.save(state, f_path)

    if is_best:
        best_fpath = best_model_path
        shutil.copyfile(f_path, best_fpath)

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
@click.option('--ref_valid', default="./drive/My Drive/Mini Project/dataset/ref_valid.pkl", help="validation reference file path")
@click.option('--best_model', default="./drive/My Drive/Mini Project/best_model/model.pt", help="best model file path")
@click.option('--checkpoint_path', default="./drive/My Drive/Mini Project/checkpoint/model_ckpt.pt", help=" model check point files path")
@click.option('--seed', default=123, help="manual seed value (default=123)")
def train(**kwargs):
    print("Loading datasets...")
    train_dataset = WikiDataset(kwargs['src_train'], kwargs['tgt_train'])
    valid_dataset = WikiDataset(kwargs['src_valid'], kwargs['tgt_valid'], kwargs['ref_valid'], ref=True)
    print("Dataset loaded successfully")

    train_dl = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, collate_fn=collate_fn, shuffle=True)
    valid_dl = DataLoader(valid_dataset, batch_size=TRAIN_BATCH_SIZE, collate_fn=collate_fn, shuffle=True)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        'weight_decay_rate': 0.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=3e-5)
    start_epoch = 0
    eval_loss = float("inf")

    if os.path.exists(kwargs["checkpoint_path"]):
        optimizer, eval_loss, start_epoch = load_checkpt(kwargs["checkpoint_path"], optimizer)
        print(f"Loading model from checkpoint with start epoch: {start_epoch} and loss: {eval_loss}")
        logging.info(f"Model loaded from saved checkpoint with start epoch: {start_epoch} and loss: {eval_loss}")
    

    train_model(start_epoch, eval_loss, (train_dl, valid_dl), optimizer, kwargs["checkpoint_path"], kwargs["best_model"])

@task.command()
@click.option('--src_test', default="./drive/My Drive/Mini Project/dataset/src_test.txt", help="test source file path")
@click.option('--tgt_test', default="./drive/My Drive/Mini Project/dataset/tgt_test.txt", help="test target file path")
@click.option('--ref_test', default="./drive/My Drive/Mini Project/dataset/ref_test.pkl", help="validation reference file path")
@click.option('--best_model', default="./drive/My Drive/Mini Project/best_model/model.pt", help="best model file path")
def test(**kwargs):
    print("Testing Model module executing...")
    logging.info(f"Test module invoked.")
    # model = EncDecModel(max_token_len)
    _, _, _ = load_checkpt(kwargs["best_model"])
    print(f"Model loaded.")
    model.eval()
    test_dataset = WikiDataset(kwargs['src_test'], kwargs['tgt_test'], kwargs['ref_test'], ref=True)
    test_dl = DataLoader(test_dataset, batch_size=TRAIN_BATCH_SIZE, collate_fn=collate_fn, shuffle=True)
    test_start_time = time.time()
    test_loss, bleu_score, sari_score = evaluate(test_dl, 0)
    test_loss = test_loss/TRAIN_BATCH_SIZE
    bleu_score = bleu_score/TRAIN_BATCH_SIZE
    sari_score = sari_score/TRAIN_BATCH_SIZE
    print(f'Avg. eval loss: {test_loss:.5f} | blue score: {bleu_score} | sari score: {sari_score} | time elapsed: {time.time() - test_start_time}')
    logging.info(f'Avg. eval loss: {test_loss:.5f} | blue score: {bleu_score} | sari score: {sari_score} | time elapsed: {time.time() - test_start_time}')
    print("Test Complete!")

@task.command()
@click.option('--src_file', default="./drive/My Drive/Mini Project/dataset/src_file.txt", help="test source file path")
@click.option('--best_model', default="./drive/My Drive/Mini Project/best_model/model.pt", help="best model file path")
@click.option('--output', default="./drive/My Drive/Mini Project/outputs/decoded.txt", help="file path to save predictions")
def decode(**kwargs):
    print("Decoding sentences module executing...")
    logging.info(f"Decode module invoked.")
    _, _, _ = load_checkpt(kwargs["best_model"])
    print(f"Model loaded.")
    model.eval()
    dataset = WikiDataset(kwargs['src_file'])
    predicted_list = []
    sent_tensors = tokenizer.encode_sent(dataset.src)
    print("Decoding Sentences...")
    for sent in sent_tensors:
        print(f"input: {sent[0].size()}")
        predicted = model.generate(sent[0].to(device), attention_mask=sent[1].to(device), decoder_start_token_id=model.config.decoder.decoder_start_token_id)
        print(f'output: {predicted.squeeze().size()}')
        predicted_list.append(predicted.squeeze())
    
    output = tokenizer.decode_sent_tokens(predicted_list)
    with open(kwargs["output"], "w") as f:
        for sent in output:
            f.write(sent + "\n")
    print("Output file saved successfully.")


def train_model(start_epoch, eval_loss, loaders, optimizer, check_pt_path, best_model_path):
    best_eval_loss = eval_loss
    print("Model training started...")
    for epoch in range(start_epoch, N_EPOCH):
        print(f"Epoch {epoch} running...")
        epoch_start_time = time.time()
        epoch_train_loss = 0
        epoch_eval_loss = 0
        model.train()
        for step, batch in enumerate(loaders[0]):
            src_tensors, src_attn_tensors, tgt_tensors, tgt_attn_tensors, labels = tokenizer.encode_batch(batch)
            optimizer.zero_grad()
            model.zero_grad()
            loss = model(input_ids = src_tensors.to(device), 
                            decoder_input_ids = tgt_tensors.to(device),
                            attention_mask = src_attn_tensors.to(device),
                            decoder_attention_mask = tgt_attn_tensors.to(device),
                            labels = labels.to(device))[0]
            if step == 0:
                epoch_train_loss = loss.item()
            else:
                epoch_train_loss = (1/2.0)*(epoch_train_loss + loss.item())
            
            loss.backward()
            optimizer.step()

            if (step+1) % LOG_EVERY == 0:
                print(f'Epoch: {epoch} | iter: {step+1} | avg. train loss: {epoch_train_loss} | time elapsed: {time.time() - epoch_start_time}')
                logging.info(f'Epoch: {epoch} | iter: {step+1} | avg. train loss: {epoch_train_loss} | time elapsed: {time.time() - epoch_start_time}')
        
        eval_start_time = time.time()
        epoch_eval_loss, bleu_score, sari_score = evaluate(loaders[1], epoch_eval_loss)
        epoch_eval_loss = epoch_eval_loss
        print(f'Completed Epoch: {epoch} | avg. eval loss: {epoch_eval_loss:.5f} | blue score: {bleu_score} | Sari score: {sari_score} | time elapsed: {time.time() - eval_start_time}')
        logging.info(f'Completed Epoch: {epoch} | avg. eval loss: {epoch_eval_loss:.5f} | blue score: {bleu_score}| Sari score: {sari_score} | time elapsed: {time.time() - eval_start_time}')

        check_pt = {
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'eval_loss': epoch_eval_loss,
            'sari_score': sari_score,
            'bleu_score': bleu_score
        }
        check_pt_time = time.time()
        print("Saving Checkpoint.......")
        if epoch_eval_loss < best_eval_loss:
            print("New best model found")
            logging.info(f"New best model found")
            best_eval_loss = epoch_eval_loss
            save_model_checkpt(check_pt, True, check_pt_path, best_model_path)
        else:
            save_model_checkpt(check_pt, False, check_pt_path, best_model_path)  
        print(f"Checkpoint saved successfully with time: {time.time() - check_pt_time}")
        logging.info(f"Checkpoint saved successfully with time: {time.time() - check_pt_time}")

        gc.collect()
        torch.cuda.empty_cache()  

    

if __name__ == "__main__":
    task()