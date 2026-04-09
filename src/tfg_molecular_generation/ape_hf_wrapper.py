import os
import json
import re
from transformers import PreTrainedTokenizerFast

class APEHuggingFaceTokenizer(PreTrainedTokenizerFast):
    """
    Fast Wrapper that uses the Rust BPE engine under the hood equipped 
    with the Unicode Translation Trick to support Molecular APE Tokenization.
    """
    def __init__(
        self,
        ape_tokenizer_path,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        **kwargs
    ):
        tokenizer_file = os.path.join(ape_tokenizer_path, "tokenizer.json")
        mapping_file = os.path.join(ape_tokenizer_path, "unicode_mapping.json")

        if not os.path.exists(tokenizer_file) or not os.path.exists(mapping_file):
            raise ValueError(f"Could not find Tokenizer or Mapping files in {ape_tokenizer_path}")

        with open(mapping_file, "r") as f:
            mapping = json.load(f)
            self.token_to_unicode = mapping["token_to_unicode"]
            self.unicode_to_token = mapping["unicode_to_token"]

        self.pattern = re.compile(r"\[[^\]]+\]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9]")

        super().__init__(
            tokenizer_file=tokenizer_file,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            mask_token=mask_token,
            **kwargs,
        )

    def translate_to_unicode(self, text):
        if not text:
            return text
        parts = self.pattern.findall(text)
        return "".join(self.token_to_unicode.get(p, p) for p in parts)

    def decode_from_unicode(self, text):
        if not text:
            return text
        return "".join(self.unicode_to_token.get(c, c) for c in text)

    # We intercept token generation calls to apply the unicode trick seamlessly
    def encode(self, text, *args, **kwargs):
        if isinstance(text, str):
            text = self.translate_to_unicode(text)
        return super().encode(text, *args, **kwargs)

    def encode_plus(self, text, *args, **kwargs):
        if isinstance(text, str):
            text = self.translate_to_unicode(text)
        return super().encode_plus(text, *args, **kwargs)

    def batch_encode_plus(self, batch_text_or_text_pairs, *args, **kwargs):
        if isinstance(batch_text_or_text_pairs, list) and isinstance(batch_text_or_text_pairs[0], str):
            batch_text_or_text_pairs = [self.translate_to_unicode(t) for t in batch_text_or_text_pairs]
        return super().batch_encode_plus(batch_text_or_text_pairs, *args, **kwargs)

    def __call__(self, text=None, *args, **kwargs):
        if text is not None:
            if isinstance(text, str):
                text = self.translate_to_unicode(text)
            elif isinstance(text, list) and len(text) > 0 and isinstance(text[0], str):
                text = [self.translate_to_unicode(t) for t in text]
                
        if "text_target" in kwargs and kwargs["text_target"] is not None:
            tt = kwargs["text_target"]
            if isinstance(tt, str):
                kwargs["text_target"] = self.translate_to_unicode(tt)
            elif isinstance(tt, list) and len(tt) > 0 and isinstance(tt[0], str):
                kwargs["text_target"] = [self.translate_to_unicode(t) for t in tt]
                
        if text is not None:
            return super().__call__(text, *args, **kwargs)
        return super().__call__(*args, **kwargs)

    # Intercept return decodes
    def decode(self, token_ids, *args, **kwargs):
        decoded_unicode = super().decode(token_ids, *args, **kwargs)
        return self.decode_from_unicode(decoded_unicode)

    def batch_decode(self, sequences, *args, **kwargs):
        decoded_batch = super().batch_decode(sequences, *args, **kwargs)
        return [self.decode_from_unicode(d) for d in decoded_batch]
