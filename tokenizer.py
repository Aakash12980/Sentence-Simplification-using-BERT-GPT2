from transformers import BertTokenizer, GPT2Tokenizer
import tqdm

class Tokenizer():
    def __init__(self, max_len=80):
        self.max_len = max_len
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.tokenizer.bos_token = self.tokenizer.cls_token
        self.tokenizer.eos_token = self.tokenizer.sep_token

        def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
            outputs = [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
            return outputs

        GPT2Tokenizer.build_inputs_with_special_tokens = build_inputs_with_special_tokens
        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.gpt2_tokenizer.pad_token = self.gpt2_tokenizer.unk_token

    def encode_batch(self, batch):
        src_tokens = self.tokenizer(batch[0], max_length = self.max_len, add_special_tokens=True,
                return_token_type_ids=False, padding="max_length", truncation=True,
                return_attention_mask=True, return_tensors="pt")

        tgt_tokens = self.gpt2_tokenizer(batch[1], max_length = self.max_len, add_special_tokens=True,
            return_token_type_ids=False, padding="max_length", truncation=True,
            return_attention_mask=True, return_tensors="pt")

        labels = tgt_tokens.input_ids.clone()
        labels[tgt_tokens.attention_mask == 0] = -100
        
        return src_tokens.input_ids, src_tokens.attention_mask, tgt_tokens.input_ids, tgt_tokens.attention_mask, labels

    def encode_sent(self, sents):
        src_tokens = []
        for s in sents:
            tokens = self.tokenizer(s, max_length=self.max_len, add_special_tokens=True,
                    return_token_type_ids=False, truncation=True, padding="max_length",
                    return_attention_mask=True, return_tensors="pt")
            src_tokens.append([tokens.input_ids, tokens.attention_mask])

        return src_tokens

    def decode_sent_tokens(self, data):
        sents_list = []
        for sent in data:
            s = self.gpt2_tokenizer.decode(sent, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            sents_list.append(s)

        return sents_list

    @staticmethod
    def get_sent_tokens(sents):
        tokenizer = Tokenizer()
        ref = []
        tokens = tokenizer.gpt2_tokenizer(sents, add_special_tokens=True,
                    return_token_type_ids=False, truncation=True, padding="longest",
                    return_attention_mask=False, return_tensors="pt")
            
        for tok in tokens.input_ids.tolist():
            ref.append([tok])

        return ref