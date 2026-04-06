import os
import json
from transformers import PreTrainedTokenizer
from tfg_molecular_generation.ape_tokenizer import APETokenizer

class APEHuggingFaceTokenizer(PreTrainedTokenizer):
    """
    Wrapper to make the custom APETokenizer compatible with HuggingFace's Trainer 
    and T5 architectures.
    """
    
    def __init__(
        self,
        ape_tokenizer_path=None,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        **kwargs
    ):
        # 1. Initiatilize the original APETokenizer
        if ape_tokenizer_path and os.path.exists(ape_tokenizer_path):
            self.ape = APETokenizer.from_pretrained(ape_tokenizer_path)
            # Update tokens based on what was loaded
            bos_token = self.ape.bos_token
            eos_token = self.ape.eos_token
            unk_token = self.ape.unk_token
            pad_token = self.ape.pad_token
            mask_token = self.ape.mask_token
        else:
            self.ape = APETokenizer(
                bos_token=bos_token,
                eos_token=eos_token,
                unk_token=unk_token,
                pad_token=pad_token,
                mask_token=mask_token
            )
            
        # We need to explicitly set vocab_file here for HF validation later
        self.vocab_file = None if ape_tokenizer_path is None else os.path.join(ape_tokenizer_path, "vocab.json")
        
        # 2. Call the HuggingFace Parent Constructor
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            mask_token=mask_token,
            **kwargs,
        )

    @property
    def vocab_size(self):
        return len(self.ape.vocabulary)

    def get_vocab(self):
        return dict(self.ape.vocabulary)

    def _tokenize(self, text):
        """
        Takes a string and converts it to a list of str tokens.
        APETokenizer encodes straight to ID's so we have to reverse it for this HF step.
        """
        # Encode returns integers (IDs) (we pass add_special_tokens=False because HF handles CLS/SEP automatically later)
        ids = self.ape.encode(text, add_special_tokens=False)
        # Convert those IDs back to tokens using self.ape method
        tokens = self.ape.convert_ids_to_tokens(ids)
        return tokens

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        # Convert single token
        return self.ape.convert_tokens_to_ids([token])[0]

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.ape.convert_ids_to_tokens([index])[0]

    def save_vocabulary(self, save_directory, filename_prefix=None):
        if not os.path.isdir(save_directory):
            os.makedirs(save_directory)
            
        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + "vocab.json"
        )
        
        # Save the pure json vocabulary for Huggingface standard
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(self.ape.vocabulary, f, ensure_ascii=False, indent=4)
            
        # Also save the APE format specifics (training state, special tokens...)
        self.ape.save_pretrained(save_directory)
        
        return (vocab_file,)
