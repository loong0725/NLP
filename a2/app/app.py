import dash
from dash import html, dcc, Input, Output, State
import torch
import torch.nn as nn
from torchtext.data.utils import get_tokenizer
import pickle

# GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Vocabulary

with open('../model/vocab_lm.pkl', 'rb') as f:
    vocab = pickle.load(f)

# Tokenizer
tokenizer = get_tokenizer('basic_english')

# Load Model
class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, num_layers, dropout_rate):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hid_dim, num_layers=num_layers, dropout=dropout_rate, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hid_dim, vocab_size)

    def init_hidden(self, batch_size, device):
        hidden = torch.zeros(num_layers, batch_size, hid_dim).to(device)
        cell = torch.zeros(num_layers, batch_size, hid_dim).to(device)
        return hidden, cell

    def forward(self, src, hidden):
        embedding = self.dropout(self.embedding(src))
        output, hidden = self.lstm(embedding, hidden)
        output = self.dropout(output)
        prediction = self.fc(output)
        return prediction, hidden

# Model Hyperparameters
vocab_size = len(vocab)
emb_dim = 1024
hid_dim = 1024
num_layers = 2
dropout_rate = 0.65

model = LSTMLanguageModel(vocab_size, emb_dim, hid_dim, num_layers, dropout_rate).to(device)
model.load_state_dict(torch.load('../model/best-val-lstm_lm.pt', map_location=device))
model.eval()

# Generate Function
def generate(prompt, max_seq_len, temperature, model, tokenizer, vocab, device):
    tokens = tokenizer(prompt)
    indices = [vocab[token] if token in vocab else vocab['<unk>'] for token in tokens]
    hidden = model.init_hidden(1, device)
    with torch.no_grad():
        for _ in range(max_seq_len):
            src = torch.LongTensor([indices]).to(device)
            prediction, hidden = model(src, hidden)
            probs = torch.softmax(prediction[:, -1] / temperature, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            if next_token == vocab['<eos>']:
                break
            indices.append(next_token)
    return ' '.join([vocab.get_itos()[i] for i in indices])

# Dash App
app = dash.Dash(__name__)
app.layout = html.Div([
    html.Div([
        html.H1("Language Model", style={"textAlign": "center"}),
        dcc.Input(id="input-prompt", type="text", placeholder="Enter a text prompt", style={"width": "40%", "margin": "10px auto", "display": "block"}),
        html.Button("Generate", id="generate-button", n_clicks=0, style={"backgroundColor": "blue", "color": "white", "border": "none", "padding": "10px 20px", "margin": "10px auto", "display": "block", "cursor": "pointer"}),
        html.Div(id="output-text", style={"marginTop": "20px", "fontSize": "18px", "textAlign": "center"})
    ], style={"display": "flex", "flexDirection": "column", "alignItems": "center", "justifyContent": "center", "height": "100vh"})
])

@app.callback(
    Output("output-text", "children"),
    Input("generate-button", "n_clicks"),
    State("input-prompt", "value")
)
def update_output(n_clicks, prompt):
    if not prompt:
        return "Please enter a prompt to generate text."
    temperatures = [0.5, 0.7, 0.75, 0.8, 1.0]
    outputs = []
    for temp in temperatures:
        continuation = generate(prompt, max_seq_len=30, temperature=temp, model=model, tokenizer=tokenizer, vocab=vocab, device=device)
        outputs.append(continuation)
    return html.Div([
        html.Div(output, style={"marginBottom": "10px"}) for output in outputs
    ])

if __name__ == "__main__":
    app.run_server(debug=True)