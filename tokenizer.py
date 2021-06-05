from transformers import BertTokenizer
import tqdm

class Tokenizer():
    def __init__(self, max_len=80):
        self.max_len = max_len
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.tokenizer.bos_token = self.tokenizer.cls_token
        self.tokenizer.eos_token = self.tokenizer.sep_token

    def encode_batch(self, batch):
        src_tokens = self.tokenizer(batch[0], add_special_tokens=True,
                return_token_type_ids=False, padding="longest", truncation=True,
                return_attention_mask=True, return_tensors="pt")
        
        tgt_tokens = self.tokenizer(batch[0], add_special_tokens=True,
                return_token_type_ids=False, padding="longest", truncation=True,
                return_attention_mask=True, return_tensors="pt")

        labels = tgt_tokens.input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return src_tokens.input_ids, src_tokens.attention_mask, tgt_tokens.input_ids, tgt_tokens.attention_mask, labels

    def encode_sent(self, sents):
        src_tokens = []
        for s in sents:
            tokens = self.tokenizer(s, max_length=self.max_len, add_special_tokens=True,
                    return_token_type_ids=False, truncation=True, padding="max_length",
                    return_attention_mask=True, return_tensors="pt")
            src_tokens.append([tokens.input_ids, tokens.attention_mask])

        return src_tokens

    @staticmethod
    def decode_sent_tokens(data):
        tokenizer_obj = Tokenizer()
        sents_list = []
        for sent in data:
            s = tokenizer_obj.tokenizer.decode(sent, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            sents_list.append(s)
        return sents_list

    @staticmethod
    def get_sent_tokens(sents):
        tokenizer = Tokenizer()
        ref = []
        tokens = tokenizer.tokenizer(sents, add_special_tokens=True,
                    return_token_type_ids=False, truncation=True, padding="longest",
                    return_attention_mask=False, return_tensors="pt")
            
        for tok in tokens.input_ids.tolist():
            ref.append([tok])

        return ref



# x = ["Hello there", "I am here to tell you this."]
# tokenizer = Tokenizer(12)
# out= tokenizer.encode_sent(x)
# print(out)