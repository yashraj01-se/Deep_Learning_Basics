import regex as re

GPT4_SPLIT_PATTERN = (
    r"'(?i:[sdmt]|ll|ve|re)"
    r"|[^\r\n\p{L}\p{N}]?\p{L}+"
    r"|\p{N}{1,3}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]"
    r"|\s+(?!\S)"
    r"|\s+"
)

GPT4_SPECIAL_TOKENS = {
    "<|endoftext|>": 100257,
    "<|fim_prefix|>": 100258,
    "<|fim_middle|>": 100259,
    "<|fim_suffix|>": 100260,
    "<|endofprompt|>": 100276,
}

gpt4pat=re.compile(GPT4_SPLIT_PATTERN)

textt = "Vellore Institute of Technology or VIT is a private deemed university[2][3] in Vellore, Tamil Nadu, India. The institution offers undergraduate and postgraduate programmes.[4] It has campuses in Vellore and Chennai and three sister universities as distinct state private universities in Amaravati,[5] Bhopal,[6] Bengaluru[7] in India and an international campus in Mauritius.[8] History: VIT was established under Section 3 of the University Grants Commission Act, 1956, and was founded in 1984 by G. Viswanathan as a self-financing institution called the Vellore Engineering College. In 2001, it became a deemed university.[9] In September 2006, it was renamed to VIT University. What truly sets VIT apart is its student-centric approach, including the Fully Flexible Credit System (FFCS), allowing students to choose their courses, faculty, and class schedules, combined with a strong research ecosystem and global partnerships that have helped VIT consistently rank among the top private universities in India and gain recognition in QS and Times Higher Education global rankings."

class GPT_Tokenizer:
    def __init__(self):
        self.merges={}
        self.vocab={}

        # Special Tokens:
        self.special_tokens_to_id=GPT4_SPECIAL_TOKENS.copy()
        self.ids_to_special_token={
            v:k for k,v in GPT4_SPECIAL_TOKENS.items()
        }

    def map_ele(self,ids):
        count = {}
        for pair in zip(ids, ids[1:]):
            count[pair] = count.get(pair, 0) + 1
        return count

    def ele_rep(self,ids, pair, idx):
        newids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids
    
    def train(self,text,vocab_size):
        chunks=re.findall(gpt4pat,text)
        ids=[]
        for chunk in chunks:
            ids.extend(list(chunk.encode("utf-8")))

        original_length = len(ids)
        num_range = vocab_size - 256
        for i in range(num_range):
            stats = self.map_ele(ids)
            if not stats:
                break
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = self.ele_rep(ids, pair, idx)
            self.merges[pair] = idx

        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]

    # ENCODING
    def encoding(self, text, allowed_special=None):
        tokens = []
        i = 0
        
        while i < len(text):
        # Check if special token starts here
            matched_special = None
            for special in self.special_tokens_to_id:
                if text.startswith(special, i):
                    matched_special = special
                    break
            
            if matched_special:
                if allowed_special != "all":
                    raise ValueError(f"Special token {matched_special} not allowed.")
                
                tokens.append(self.special_tokens_to_id[matched_special])
                i += len(matched_special)
                continue

        # Otherwise process normal segment until next special token
            next_special_positions = [text.find(s, i)
                                    for s in self.special_tokens_to_id
                                    if text.find(s, i) != -1]

            next_cut = min(next_special_positions) if next_special_positions else len(text)
            segment = text[i:next_cut]
            chunks = re.findall(gpt4pat, segment)

            for chunk in chunks:
                ids = list(chunk.encode("utf-8"))

                while True:
                    stats = self.map_ele(ids)
                    if not stats:
                        break

                    pair = min(
                        stats,
                        key=lambda p: self.merges.get(p, float("inf"))
                    )

                    if pair not in self.merges:
                        break

                    idx = self.merges[pair]
                    ids = self.ele_rep(ids, pair, idx)

                tokens.extend(ids)

            i = next_cut

        return tokens
    
    # DECODING
    def decoding(self,ids):
        output=[]
        for idx in ids:
            if idx in self.ids_to_special_token:
                output.append(self.ids_to_special_token[idx])
            else:
                output.append(
                    self.vocab[idx].decode("utf-8",errors="replace")
                )
        
        return "".join(output)
               
########################################################################################################

tokenizer = GPT_Tokenizer()

# Train on small sample
tokenizer.train("How are you? 123!!!", vocab_size=300)

# Normal encoding
encoded = tokenizer.encoding("How are you? 123!!!")
decoded = tokenizer.decoding(encoded)

print("Encoded:", encoded)
print("Decoded:", decoded)
print("Round-trip correct:", decoded == "How are you? 123!!!")

print("\n--- Special Token Test ---")

encoded_special = tokenizer.encoding(
    "<|endoftext|>Hello world",
    allowed_special="all"
)

decoded_special = tokenizer.decoding(encoded_special)

print("Encoded with special:", encoded_special)
print("Decoded with special:", decoded_special)

########################################################################################################