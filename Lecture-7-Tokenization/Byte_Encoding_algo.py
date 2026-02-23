textt = "Vellore Institute of Technology or VIT is a private deemed university[2][3] in Vellore, Tamil Nadu, India. The institution offers undergraduate and postgraduate programmes.[4] It has campuses in Vellore and Chennai and three sister universities as distinct state private universities in Amaravati,[5] Bhopal,[6] Bengaluru[7] in India and an international campus in Mauritius.[8] History: VIT was established under Section 3 of the University Grants Commission Act, 1956, and was founded in 1984 by G. Viswanathan as a self-financing institution called the Vellore Engineering College. In 2001, it became a deemed university.[9] In September 2006, it was renamed to VIT University. What truly sets VIT apart is its student-centric approach, including the Fully Flexible Credit System (FFCS), allowing students to choose their courses, faculty, and class schedules, combined with a strong research ecosystem and global partnerships that have helped VIT consistently rank among the top private universities in India and gain recognition in QS and Times Higher Education global rankings."

token = textt.encode("utf-8")

def map_ele(ids):
    count = {}
    for pair in zip(ids, ids[1:]):
        count[pair] = count.get(pair, 0) + 1
    return count

def ele_rep(ids, pair, idx):
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

# TRAINING
vocab_size = 276
num_range = vocab_size - 256

ids = list(token)
original_length = len(ids)

merges = {}

for i in range(num_range):
    stats = map_ele(ids)
    pair = max(stats, key=stats.get)
    idx = 256 + i

    print(f"Merging {pair} with idx: {idx}")

    ids = ele_rep(ids, pair, idx)
    merges[pair] = idx

final_length = len(ids)

print("\nOriginal length:", original_length)
print("Final length:", final_length)
print("Compression ratio:", final_length / original_length)
print("Final vocabulary size:", 256 + len(merges))

# BUILD VOCAB FOR DECODING
vocab = {idx: bytes([idx]) for idx in range(256)}
for (p0, p1), idx in merges.items():
    vocab[idx] = vocab[p0] + vocab[p1]

# DECODING
def decoding(ids):
    token_list = b"".join(vocab[idx] for idx in ids)
    decoded_text = token_list.decode("utf-8", errors="replace")
    return decoded_text

# ENCODING
def encoding(text):
    token = list(text.encode("utf-8"))

    while True:
        stats = map_ele(token)

        pair = min(
            stats,
            key=lambda p: merges.get(p, float("inf"))
        )

        if pair not in merges:
            break

        idx = merges[pair]
        token = ele_rep(token, pair, idx)

    return token

# TEST
test_text = "Hello world!"
encoded = encoding(test_text)
decoded = decoding(encoded)

print("\nTest text:", test_text)
print("Encoded:", encoded)
print("Decoded:", decoded)
print("Round-trip correct:", decoded == test_text)