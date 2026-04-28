"""
Just a simple character level tokenizer.

From: https://github.com/dariush-bahrami/character-tokenizer/blob/master/charactertokenizer/core.py

CharacterTokenzier for Hugging Face Transformers.
This is heavily inspired from CanineTokenizer in transformers package.
"""

import json
import os
import random
import torch
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union, Tuple
from torch.utils.data import Sampler

from transformers.tokenization_utils import AddedToken, PreTrainedTokenizer


class CharacterTokenizer(PreTrainedTokenizer):
    def __init__(
        self,
        characters: Sequence[str],
        model_max_length: int,
        padding_side: str = "left",
        **kwargs
    ):
        """Character tokenizer for Hugging Face transformers.
        Args:
            characters (Sequence[str]): List of desired characters. Any character which
                is not included in this list will be replaced by a special token called
                [UNK] with id=6. Following are list of all of the special tokens with
                their corresponding ids:
                    "[CLS]": 0
                    "[SEP]": 1
                    "[BOS]": 2
                    "[MASK]": 3
                    "[PAD]": 4
                    "[RESERVED]": 5
                    "[UNK]": 6
                an id (starting at 7) will be assigned to each character.
            model_max_length (int): Model maximum sequence length.
        """
        self.characters = characters
        self.model_max_length = model_max_length
        bos_token = AddedToken("[BOS]", lstrip=False, rstrip=False)
        eos_token = AddedToken("[SEP]", lstrip=False, rstrip=False)
        sep_token = AddedToken("[SEP]", lstrip=False, rstrip=False)
        cls_token = AddedToken("[CLS]", lstrip=False, rstrip=False)
        pad_token = AddedToken("[PAD]", lstrip=False, rstrip=False)
        unk_token = AddedToken("[UNK]", lstrip=False, rstrip=False)

        mask_token = AddedToken("[MASK]", lstrip=True, rstrip=False)

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            unk_token=unk_token,
            add_prefix_space=False,
            model_max_length=model_max_length,
            padding_side=padding_side,
            **kwargs,
        )

        self._vocab_str_to_int = {
            "[CLS]": 0,
            "[SEP]": 1,
            "[BOS]": 2,
            "[MASK]": 3,
            "[PAD]": 4,
            "[RESERVED]": 5,
            "[UNK]": 6,
            **{ch: i + 7 for i, ch in enumerate(characters)},
        }
        self._vocab_int_to_str = {v: k for k, v in self._vocab_str_to_int.items()}

    @property
    def vocab_size(self) -> int:
        return len(self._vocab_str_to_int)
    
    def _tokenize(self, text: str) -> List[str]:
        return list(text)

    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab_str_to_int.get(token, self._vocab_str_to_int["[UNK]"])

    def _convert_id_to_token(self, index: int) -> str:
        return self._vocab_int_to_str[index]

    def get_vocab(self) -> Dict[str, int]:
        """Return the vocabulary as a dictionary mapping token strings to token ids."""
        return self._vocab_str_to_int.copy()

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        result = cls + token_ids_0 + sep
        if token_ids_1 is not None:
            result += token_ids_1 + sep
        return result

    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False,
    ) -> List[int]:
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0,
                token_ids_1=token_ids_1,
                already_has_special_tokens=True,
            )

        result = [1] + ([0] * len(token_ids_0)) + [1]
        if token_ids_1 is not None:
            result += ([0] * len(token_ids_1)) + [1]
        return result

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        result = len(cls + token_ids_0 + sep) * [0]
        if token_ids_1 is not None:
            result += len(token_ids_1 + sep) * [1]
        return result

    def get_config(self) -> Dict:
        return {
            "char_ords": [ord(ch) for ch in self.characters],
            "model_max_length": self.model_max_length,
        }

    @classmethod
    def from_config(cls, config: Dict) -> "CharacterTokenizer":
        cfg = {} 
        cfg["characters"] = [chr(i) for i in config["char_ords"]]
        cfg["model_max_length"] = config["model_max_length"]
        return cls(**cfg)

    def save_pretrained(self, save_directory: Union[str, os.PathLike], **kwargs):
        cfg_file = Path(save_directory) / "tokenizer_config.json"
        cfg = self.get_config()
        with open(cfg_file, "w") as f:
            json.dump(cfg, f, indent=4)

    @classmethod
    def from_pretrained(cls, save_directory: Union[str, os.PathLike], **kwargs):
        cfg_file = Path(save_directory) / "tokenizer_config.json"
        with open(cfg_file) as f:
            cfg = json.load(f)
        return cls.from_config(cfg)

class BatchStratifiedSampler(Sampler):
    def __init__(self, labels, batch_size):
        assert batch_size % 2 == 0
        self.pos_label_idx = [i for i,l in enumerate(labels) if l == 1]
        self.neg_label_idx = [i for i,l in enumerate(labels) if l == 0]
        self.half = batch_size // 2

    def __iter__(self):
        random.shuffle(self.pos_label_idx)
        random.shuffle(self.neg_label_idx)
        for i in range(0, len(self.pos_label_idx), self.half):
            pos_batch = self.pos_label_idx[i:i+self.half]
            neg_batch = self.neg_label_idx[i:i+self.half]
            if len(neg_batch) < self.half:
                neg_batch = neg_batch + random.sample(self.neg_label_idx, self.half - len(neg_batch))
            # Combine the positive and negative index batches
            batch = pos_batch + neg_batch
            # Shuffle the final batch to mix positive and negative samples
            random.shuffle(batch)
            yield batch

    def __len__(self):
        return len(self.pos_label_idx) // self.half


# helper functions
def exists(val):
    return val is not None


def coin_flip():
    return random() > 0.5


string_complement_map = {
    "A": "U",
    "A": "T",
    "C": "G",
    "G": "C",
    "T": "A",
    "a": "t",
    "a": "u",
    "c": "g",
    "g": "c",
    "t": "a",
}

# augmentation
def string_reverse_complement(seq):
    rev_comp = ""
    for base in seq[::-1]:
        if base in string_complement_map:
            rev_comp += string_complement_map[base]
        # if bp not complement map, use the same bp
        else:
            rev_comp += base
    return rev_comp

class miRawDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataframe,
        mRNA_max_length=40,
        miRNA_max_length=26,
        mRNA_col="mRNA sequence",
        miRNA_col="miRNA sequence",
        seed_start_col=None,
        seed_end_col=None,
        tokenizer=None,
        use_padding=None,
        rc_aug=None,
        add_eos=False,
        concat=False,
        add_linker=False,
    ):
        self.use_padding = use_padding
        self.mRNA_max_length = mRNA_max_length
        self.miRNA_max_length = miRNA_max_length
        self.tokenizer = tokenizer
        self.rc_aug = rc_aug
        self.add_eos = add_eos
        self.concat = concat
        self.data = dataframe
        self.add_linker = add_linker
        self.mRNA_sequences = self.data[[mRNA_col]].values
        self.miRNA_sequences = self.data[[miRNA_col]].values
        self.labels = self.data[["label"]].values
        if seed_start_col is not None and seed_end_col is not None:
            self.seed_starts = self.data[[seed_start_col]].values
            self.seed_ends = self.data[[seed_end_col]].values
        else:
            self.seed_starts = None
            self.seed_ends = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Convert data to PyTorch tensors
        mRNA_seq = self.mRNA_sequences[idx][0]
        miRNA_seq = self.miRNA_sequences[idx][0]
        label = self.labels[idx]
        if self.seed_starts is not None and self.seed_ends is not None:
            seed_start = self.seed_starts[idx][0]
            seed_end = self.seed_ends[idx][0]
        else:
            seed_start = None
            seed_end = None       
        # replace U with T and reverse miRNA from 5'-to-3' to 3'-to-5'
        miRNA_seq = miRNA_seq.replace("U", "T")[::-1]

        # apply rc_aug here if using
        if self.rc_aug and coin_flip():
            mRNA_seq = string_reverse_complement(mRNA_seq)
            miRNA_seq = string_reverse_complement(miRNA_seq)

        mRNA_seq_encoding = self.tokenizer(
            mRNA_seq,
            add_special_tokens=False,
            padding="max_length" if self.use_padding else "longest", # default to padding to longest in the batch
            max_length=self.mRNA_max_length,
            truncation=True,
            return_attention_mask=True,
        )

        mRNA_seq_tokens = mRNA_seq_encoding["input_ids"]  # get input_ids
        mRNA_seq_mask = mRNA_seq_encoding["attention_mask"]  # get attention mask
        # get padded length and add the length to seed start and seed end
        pad_token_id = self.tokenizer.pad_token_id
        pad_token_length = mRNA_seq_tokens.count(pad_token_id)
        seed_start = seed_start + pad_token_length if seed_start > 0 else seed_start
        seed_end = seed_end + pad_token_length if seed_end > 0 else seed_end

        miRNA_seq_encoding = self.tokenizer(
            miRNA_seq,
            add_special_tokens=False,
            padding="max_length" if self.use_padding else "longest",
            max_length=self.miRNA_max_length,
            truncation=True,
            return_attention_mask=True,
        )
        miRNA_seq_tokens = miRNA_seq_encoding["input_ids"]  # get input_ids
        miRNA_seq_mask = miRNA_seq_encoding["attention_mask"]  # get attention mask

        # need to handle eos here
        if self.add_eos:
            # append list seems to be faster than append tensor
            mRNA_seq_tokens.append(self.tokenizer.sep_token_id)
            miRNA_seq_tokens.append(self.tokenizer.sep_token_id)
            mRNA_seq_mask.append(1)  # do not mask eos token
            miRNA_seq_mask.append(1)  # do not mask eos token
        
        if self.concat:
            if self.add_linker:
                linker = [self.tokenizer._convert_token_to_id("N")] * 6
                concat_seq_tokens = mRNA_seq_tokens + linker + miRNA_seq_tokens
                linker_mask = [1] * 6
                concat_seq_mask = mRNA_seq_mask + linker_mask + miRNA_seq_mask
            else:
                # concatenate miRNA and mRNA tokens
                concat_seq_tokens = mRNA_seq_tokens + miRNA_seq_tokens
                concat_seq_mask = mRNA_seq_mask + miRNA_seq_mask
            # convert to tensor
            concat_seq_tokens = torch.tensor(concat_seq_tokens, dtype=torch.long)
            concat_seq_mask = torch.tensor(concat_seq_mask, dtype=torch.long)
            target = torch.tensor([label], dtype=torch.float)
            
            return concat_seq_tokens, concat_seq_mask, seed_start, seed_end, target
        else:
            # convert to tensor
            mRNA_seq_tokens = torch.tensor(mRNA_seq_tokens, dtype=torch.long)
            miRNA_seq_tokens = torch.tensor(miRNA_seq_tokens, dtype=torch.long)
            mRNA_seq_mask = torch.tensor(mRNA_seq_mask, dtype=torch.long)
            miRNA_seq_mask = torch.tensor(miRNA_seq_mask, dtype=torch.long)
            target = torch.tensor([label], dtype=torch.float)
            return mRNA_seq_tokens, miRNA_seq_tokens, mRNA_seq_mask, miRNA_seq_mask, seed_start, seed_end, target

class SpanDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data,
                 mrna_max_len,
                 mirna_max_len,
                 tokenizer,
                 mRNA_col="mRNA sequence",
                 miRNA_col="miRNA sequence",
                 seed_start_col=None,
                 seed_end_col=None,
                 cleavage_site_col="cleave_site",
                 ):
        self.data = data
        self.mrna_max_len = mrna_max_len
        self.mirna_max_len = mirna_max_len
        self.tokenizer = tokenizer
        self.mRNA_col = mRNA_col
        self.miRNA_col = miRNA_col
        self.mRNA_sequences = self.data[[mRNA_col]].values
        self.miRNA_sequences = self.data[[miRNA_col]].values
        self.seed_start_col = seed_start_col
        self.seed_end_col   = seed_end_col
        self.cleavage_site_col = cleavage_site_col
        self.seed_starts = self.data[[seed_start_col]].values
        self.seed_ends = self.data[[seed_end_col]].values
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        mrna_seq = self.data[self.mRNA_col].iat[idx]
        mirna_seq = self.data[self.miRNA_col].iat[idx]
        label = self.data["label"].iat[idx] if "label" in self.data.columns else 1  # Default to positive if no label

        if self.seed_start_col is not None and self.seed_end_col is not None:
            seed_start = torch.tensor(self.data[self.seed_start_col].iat[idx], dtype=torch.long)
            seed_end = torch.tensor(self.data[self.seed_end_col].iat[idx], dtype=torch.long)
        else:
            seed_start = -1
            seed_end = -1
            
        # Get cleavage site position
        if self.cleavage_site_col is not None and self.cleavage_site_col in self.data.columns:
            cleavage_sites = torch.tensor(self.data[self.cleavage_site_col].iat[idx], dtype=torch.long)
            cleavage_soft_targets = self.gaussian_targets(L=self.mrna_max_len, pos=cleavage_sites) # (mrna_max_len, )
        else:
            cleavage_soft_targets = float(-1)
            cleavage_sites = -1 # no cleavage site
        
        # Tokenize mirna
        mirna_seq = mirna_seq.replace("U", "T")
        # mirna_seq = mirna_seq[::-1]
        mirna_encoded = self.tokenizer(
            mirna_seq,
            add_special_tokens=False,
            padding="max_length",
            truncation=True,
            max_length=self.mirna_max_len,
            return_attention_mask=True,
        )

        # Tokenize mrna
        mrna_encoded = self.tokenizer(
            mrna_seq,
            add_special_tokens=False,
            padding="max_length",
            truncation=True,
            max_length=self.mrna_max_len,  
            return_attention_mask=True,
        )
        
        # Convert mRNA tokenization results to tensors
        mirna_ids = torch.tensor(mirna_encoded["input_ids"], dtype=torch.long)
        mirna_attn_mask = torch.tensor(mirna_encoded["attention_mask"], dtype=torch.long)
        mrna_ids = torch.tensor(mrna_encoded["input_ids"], dtype=torch.long)  # Use modified list with global token
        mrna_attn_mask = torch.tensor(mrna_encoded["attention_mask"], dtype=torch.long)  # Use modified list with global token
        target = torch.tensor([label], dtype=torch.long)

        return {
            "mirna_input_ids": mirna_ids,
            "mirna_attention_mask": mirna_attn_mask,
            "mrna_input_ids": mrna_ids,
            "mrna_attention_mask": mrna_attn_mask,
            "start_positions": seed_start,  # used as labels
            "end_positions": seed_end,
            "target": target,
            "cleavage_sites": cleavage_sites,
            "cleavage_soft_targets": cleavage_soft_targets
        }
    
    def gaussian_targets(self, L: int, pos: int, sigma: float = 3.0):
        """
        L: Window length
        pos: True cleavage site index (0..L-1)
        sigma: Gaussian width, 30nt suggested 2~3
        Returns: shape [L] float tensor, peak=1
        """
        idx = torch.arange(L, dtype=torch.float32)
        y = torch.exp(-0.5 * ((idx - float(pos)) / float(sigma)) ** 2)
        # peak normalization, make max value=1 
        y = y / y.max()
        return y
      
class TokenClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 data, 
                 tokenizer, 
                 mrna_max_len: int,
                 mirna_max_len: int):
        """
        data: pandas.DataFrame, needs to include three columns:
            - "mRNA sequence": sequence string
            - "miRNA sequence": sequence string
            - "seeds": List[Tuple[int,int]], (start, end) of each seed region
            - "label": 0 or 1
        tokenizer: HuggingFace tokenizer that supports __call__(str, ...)
        max_len: int, pad / truncate to this length
        """
        self.tokenizer = tokenizer
        self.mrna_max_len = mrna_max_len
        self.mirna_max_len = mirna_max_len
        self.mrna_seqs = data["mRNA sequence"].tolist()
        self.mirna_seqs = data["miRNA sequence"].tolist()
        self.seed_l = data["seeds"].tolist()
        self.labels = data["label"].tolist()

    def __len__(self):
        return len(self.mrna_seqs)

    def __getitem__(self, idx):
        mrna_seq = self.mrna_seqs[idx]
        mirna_seq = self.mirna_seqs[idx]
        seeds = self.seed_l[idx]
        label = self.labels[idx]

        mrna_tok = self.tokenizer(
            mrna_seq,
            padding="max_length",
            truncation=True,
            max_length=self.mrna_max_len,
            return_attention_mask=True
        )

        mirna_tok = self.tokenizer(
            mirna_seq,
            padding="max_length",
            truncation=True,
            max_length=self.mirna_max_len,
            return_attention_mask=True
        )
        
        if label == 1:
            bio_labels = self._make_labels(mrna_seq, seeds)
        else:
            bio_labels = [-100] * self.mrna_max_len

        return {
            "mrna_input_ids":      torch.tensor(mrna_tok["input_ids"],      dtype=torch.long),
            "mrna_attention_mask": torch.tensor(mrna_tok["attention_mask"], dtype=torch.long),
            "mirna_input_ids":      torch.tensor(mirna_tok["input_ids"],      dtype=torch.long),
            "mirna_attention_mask": torch.tensor(mirna_tok["attention_mask"], dtype=torch.long),
            "labels":         torch.tensor(bio_labels, dtype=torch.long),
            "binding_labels": torch.tensor([label],    dtype=torch.float)
        }

    def _make_labels(self, seq: str, seeds: list[tuple[int,int]]):
        """
        IO tagging: O=0, I=1; padding part is ignored with -100
        seq: original sequence
        seeds: [(start,end), ...], all are indices on the sequence
        Returns a List[int] of length exactly self.max_len
        """
        lab = [0] * len(seq)
        for (s, e) in seeds:
            for i in range(s, min(e+1, len(seq))):
                lab[i] = 1
        if len(lab) >= self.mrna_max_len:
            lab = lab[:self.mrna_max_len]
        else:
            lab = lab + [-100] * (self.mrna_max_len - len(lab))
        return lab           
        
class TargetPredictionDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data,
                 mrna_max_len,
                 mirna_max_len,
                 tokenizer,
                 mRNA_col="mRNA sequence",
                 miRNA_col="miRNA sequence",):
        self.data = data
        self.mrna_max_len = mrna_max_len
        self.mirna_max_len = mirna_max_len
        self.tokenizer = tokenizer
        self.mRNA_col = mRNA_col
        self.miRNA_col = miRNA_col

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # NOTE: CharacterTokenizer defines eos_token as "[SEP]" (id=1) and does NOT have "[EOS]".
        # Using "[EOS]" would map to [UNK] and silently break generation stop conditions.
        bos = self.tokenizer.bos_token_id
        eos = self.tokenizer.eos_token_id
        pad = self.tokenizer.pad_token_id

        mrna_seq = self.data[self.mRNA_col].iat[idx]
        mirna_seq = self.data[self.miRNA_col].iat[idx]

        # Tokenize mirna
        mirna_seq = mirna_seq.replace("U", "T")
        mirna_seq = mirna_seq[::-1] # 3' to 5'
        mirna_encoded = self.tokenizer(
            mirna_seq,
            add_special_tokens=False,
            padding=False, # we will handle padding later
            truncation=True,
            max_length=self.mirna_max_len - 2, 
        )
        assert pad not in mirna_encoded["input_ids"], "Tokenizer unexpectedly padded even though padding=False"
        mirna_encoded["input_ids"] = [bos] + mirna_encoded["input_ids"] + [eos]
        # pad to fixed length
        if len(mirna_encoded["input_ids"]) < self.mirna_max_len:
            mirna_encoded["input_ids"] = mirna_encoded["input_ids"] + [pad] * (self.mirna_max_len - len(mirna_encoded["input_ids"]))
        else:
            mirna_encoded["input_ids"] = mirna_encoded["input_ids"][:self.mirna_max_len]

        # Tokenize mrna
        mrna_encoded = self.tokenizer(
            mrna_seq,
            add_special_tokens=False,
            padding=False, # we will handle padding later
            truncation=True,
            max_length=self.mrna_max_len - 1,  
        )
        assert pad not in mrna_encoded["input_ids"], "Tokenizer unexpectedly padded even though padding=False"
        mrna_encoded["input_ids"] = mrna_encoded["input_ids"] + [eos]

        # pad to fixed length
        if len(mrna_encoded["input_ids"]) < self.mrna_max_len:
            mrna_encoded["input_ids"] = mrna_encoded["input_ids"] + [pad] * (self.mrna_max_len - len(mrna_encoded["input_ids"]))
        else:
            mrna_encoded["input_ids"] = mrna_encoded["input_ids"][:self.mrna_max_len]
    
        mirna_ids = torch.tensor(mirna_encoded["input_ids"], dtype=torch.long)
        mrna_ids = torch.tensor(mrna_encoded["input_ids"], dtype=torch.long)

        return {
            "mirna_input_ids": mirna_ids,
            "mrna_input_ids": mrna_ids
        }