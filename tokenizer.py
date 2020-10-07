from transformers import BertConfig, BertTokenizer, BertModel
import torch
import pickle
from tqdm import tqdm
from dataset import WikiDataset
from torch.utils.data import DataLoader

def create_sent_tokens(src_data):
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    src_tokens = []
    for src in tqdm(src_data):
        tokens = tokenizer.encode(src, return_tensors="pt", add_special_tokens=True)
        src_tokens.append(tokens)
    
    return src_tokens

def generate_tokens(batch):
    #batch is a tuple
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    src_tokens = tokenizer(batch[0], padding=True, truncation=True, return_tensors="pt")
    tgt_tokens = tokenizer(batch[1], padding=True, truncation=True, return_tensors="pt")

    return src_tokens["input_ids"], src_tokens["attention_mask"], tgt_tokens["input_ids"], tgt_tokens["attention_mask"]
    
def get_sent_from_tokens(data_ind):
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    sents_list = []
    
    for sent in tqdm(data_ind):
        s = tokenizer.decode(sent, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        sents_list.append(s)

    return sents_list


# def get_embeddings(**kwargs):
#     if kwargs["file_path"] is not None:
#         embeddings = pickle.load(open(kwargs["file_path"], "rb"))

#     else:
#         model = BertModel.from_pretrained("bert-base-cased", output_hidden_states=True)
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#         print(f"Device: {device}")
#         model = model.to(device)
#         model.eval()
#         embeddings = []
#         for tensor in tqdm(kwargs["tokens"]):
#             tensor = tensor.to(device)
#             with torch.no_grad():
#                 outputs = model(tensor)
#             embed_tensors = outputs[0]
#             embeddings.append(embed_tensors.squeeze())
#     return embeddings

# def collate_fn(batch):
#     data_list, label_list = [], []
#     for _data, _label in batch:
#         data_list.append(_data)
#         label_list.append(_label)
#     return data_list, label_list

# x = ["My name is Aakash", "Hello there", "Oh boy", "you"]
# y = ["there you go", "yes!", "Okay or ok", "Nooooooo"]
# dataset = WikiDataset(x, y)
# dl = DataLoader(dataset, batch_size=2, collate_fn=collate_fn, shuffle=True)
# for i, batch in enumerate(dl):
#     print(generate_tokens(batch))  


