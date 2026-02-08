# app.py  —— Dash 中译英翻译系统（适配你的自定义 Transformer 模型）

import dash
from dash import html, dcc, Input, Output, State
import torch
import torch.nn as nn
import math
import jieba



# -------------------- 基本配置 --------------------
MODEL_PATH = "../models/general_Seq2SeqTransformer.pt"
VOCAB_PATH = "../models/vocab.pt"
MAX_LEN = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------- 你的原始模型结构 --------------------
class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        assert hid_dim % n_heads == 0
        self.hid_dim  = hid_dim
        self.n_heads  = n_heads
        self.head_dim = hid_dim // n_heads
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale   = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0,2,1,3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0,2,1,3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0,2,1,3)
        energy = torch.matmul(Q, K.permute(0,1,3,2)) / self.scale
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        attention = torch.softmax(energy, dim=-1)
        x = torch.matmul(self.dropout(attention), V)
        x = x.permute(0,2,1,3).contiguous()
        x = x.view(batch_size, -1, self.hid_dim)
        x = self.fc_o(x)
        return x, attention

class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        self.fc1 = nn.Linear(hid_dim, pf_dim)
        self.fc2 = nn.Linear(pf_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        return self.fc2(self.dropout(torch.relu(self.fc1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device):
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm        = nn.LayerNorm(hid_dim)
        self.self_attention       = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.feedforward          = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout              = nn.Dropout(dropout)
    def forward(self, src, src_mask):
        _src, _ = self.self_attention(src, src, src, src_mask)
        src = self.self_attn_layer_norm(src + self.dropout(_src))
        _src = self.feedforward(src)
        src = self.ff_layer_norm(src + self.dropout(_src))
        return src

class Encoder(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, device, max_length=100):
        super().__init__()
        self.device = device
        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        self.layers = nn.ModuleList([
            EncoderLayer(hid_dim, n_heads, pf_dim, dropout, device)
            for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.scale   = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
    def forward(self, src, src_mask):
        batch_size = src.shape[0]
        src_len    = src.shape[1]
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))
        for layer in self.layers:
            src = layer(src, src_mask)
        return src

class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device):
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm  = nn.LayerNorm(hid_dim)
        self.ff_layer_norm        = nn.LayerNorm(hid_dim)
        self.self_attention    = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.feedforward       = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout           = nn.Dropout(dropout)
    def forward(self, trg, enc_src, trg_mask, src_mask):
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))
        _trg = self.feedforward(trg)
        trg = self.ff_layer_norm(trg + self.dropout(_trg))
        return trg, attention

class Decoder(nn.Module):
    def __init__(self, output_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, device, max_length=100):
        super().__init__()
        self.device = device
        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        self.layers = nn.ModuleList([
            DecoderLayer(hid_dim, n_heads, pf_dim, dropout, device)
            for _ in range(n_layers)
        ])
        self.fc_out  = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale   = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
    def forward(self, trg, enc_src, trg_mask, src_mask):
        batch_size = trg.shape[0]
        trg_len    = trg.shape[1]
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size,1).to(self.device)
        trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))
        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)
        output = self.fc_out(trg)
        return output, attention

class Seq2SeqTransformer(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, trg_pad_idx, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
    def make_src_mask(self, src):
        return (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()
        return trg_pad_mask & trg_sub_mask
    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)
        return output, attention

# -------------------- 载入词表 --------------------
vocab_transform = torch.load(VOCAB_PATH, map_location=DEVICE)
SRC_LANGUAGE = 'zh'
TGT_LANGUAGE = 'en'
src_vocab = vocab_transform[SRC_LANGUAGE]
tgt_vocab = vocab_transform[TGT_LANGUAGE]

SRC_BOS = src_vocab['<sos>']
SRC_EOS = src_vocab['<eos>']
SRC_PAD = src_vocab['<pad>']

TRG_BOS = tgt_vocab['<sos>']
TRG_EOS = tgt_vocab['<eos>']
TRG_PAD = tgt_vocab['<pad>']



# -------------------- 构建模型 --------------------
INPUT_DIM = len(src_vocab)
OUTPUT_DIM = len(tgt_vocab)
HID_DIM = 256
ENC_LAYERS = 3
DEC_LAYERS = 3
ENC_HEADS = 8
DEC_HEADS = 8
ENC_PF_DIM = 512
DEC_PF_DIM = 512
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1

encoder = Encoder(INPUT_DIM, HID_DIM, ENC_LAYERS, ENC_HEADS, ENC_PF_DIM, ENC_DROPOUT, DEVICE)
decoder = Decoder(OUTPUT_DIM, HID_DIM, DEC_LAYERS, DEC_HEADS, DEC_PF_DIM, DEC_DROPOUT, DEVICE)
model = Seq2SeqTransformer(encoder, decoder, SRC_PAD, TRG_PAD, DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()
# -------------------- 翻译函数 --------------------
def translate_sentence(sentence, model, max_len=50):

    model.eval()

    tokens = list(jieba.cut(sentence.strip()))

    stoi = src_vocab.get_stoi()

    src_indexes = [stoi['<sos>']]

    for tok in tokens:
        src_indexes.append(stoi.get(tok, stoi['<unk>']))

    src_indexes.append(stoi['<eos>'])

    print("Tokens:", tokens)
    print("Indexes:", src_indexes)
    print("Max index:", max(src_indexes), "Vocab size:", len(src_vocab))

    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(DEVICE)

    src_mask = model.make_src_mask(src_tensor)

    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)

    trg_indexes = [tgt_vocab['<sos>']]

    for _ in range(max_len):

        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(DEVICE)

        trg_mask = model.make_trg_mask(trg_tensor)

        with torch.no_grad():
            output, _ = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)

        logits = output[:, -1, :]
        for prev in trg_indexes:
            logits[0, prev] /= 1.2

        pred_token = logits.argmax(1).item()

        trg_indexes.append(pred_token)

        if pred_token == tgt_vocab['<eos>']:
            break

    trg_tokens = tgt_vocab.lookup_tokens(trg_indexes)

    return " ".join(trg_tokens[1:-1])



# -------------------- Dash App --------------------
app = dash.Dash(__name__)
app.layout = html.Div([
    html.H2("中 → 英 机器翻译系统", style={'textAlign': 'center'}),
    dcc.Textarea(id='input-text', placeholder='请输入中文...', style={'width': '100%', 'height': 120}),
    html.Br(),
    html.Button('翻译', id='translate-btn', n_clicks=0),
    html.Br(), html.Br(),
    dcc.Textarea(id='output-text', placeholder='翻译结果...', style={'width': '100%', 'height': 120})
])

@app.callback(
    Output('output-text', 'value'),
    Input('translate-btn', 'n_clicks'),
    State('input-text', 'value')
)
def run_translation(n, text):
    if not text:
        return ""
    return translate_sentence(text, model)

print("PAD:", src_vocab['<pad>'])
print("BOS:", src_vocab['<bos>'])
print("EOS:", src_vocab['<eos>'])

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)
