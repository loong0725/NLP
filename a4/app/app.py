from dash import Dash, html, dcc, Input, Output, State
import numpy as np
import pickle
import torch, torchtext
import os

from transformers import BertTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from bert import BERT, calculate_similarity
from transformers import BertTokenizer

# Use GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

current_dir = os.path.dirname(os.path.abspath(__file__))

model_path = '../model/bert_2.pth'
params, state = torch.load(model_path)
model_bert = BERT(**params, device=device).to(device)
model_bert.load_state_dict(state)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

app = Dash(__name__)

# Layout
app.layout = html.Div([
    html.H1("A4: Do you Agree?", style={'textAlign': 'center', 'font-family': 'Arial, sans-serif', 'margin-top': '20px'}),
    html.Div([
        html.Div([
            dcc.Input(
                id='query-one',
                type='text',
                placeholder='Enter your sentence...',
                style={
                    'width': '70%',
                    'margin': '0 auto',
                    'padding': '10px',
                    'display': 'block'
                }
            ),
            dcc.Input(
                id='query-two',
                type='text',
                placeholder='Enter your sentence...',
                style={
                    'width': '70%',
                    'margin': '0 auto',
                    'padding': '10px',
                    'display': 'block',
                    'margin-top': '20px'
                }
            ),
            html.Button(
                'Generate',
                id='search-button',
                n_clicks=0,
                style={
                    'padding': '10px 20px',
                    'background-color': '#007BFF',
                    'color': 'white',
                    'border': 'none',
                    'border-radius': '5px',
                    'margin-top': '20px',
                    'display': 'block',
                    'margin-left': 'auto',
                    'margin-right': 'auto'
                }
            ),
        ], style={
            'textAlign': 'center',
            'padding': '20px',
            'background-color': '#f9f9f9',
            'border': '1px solid #e0e0e0',
            'border-radius': '10px',
            'box-shadow': '0 4px 8px rgba(0, 0, 0, 0.1)',
            'width': '40%',
            'margin': '0 auto'
        }),
    ], style={'margin-top': '40px'}),
    html.Div(
        id='search-results',
        style={
            'margin-top': '40px',
            'padding': '20px',
            'font-family': 'Arial, sans-serif',
            'display': 'flex',
            'justify-content': 'center',
        }
    ),
])

mapping = {

}

# Callback to handle search queries
@app.callback(
    Output('search-results', 'children'),
    [Input('search-button', 'n_clicks')],
    [State('query-one', 'value')],
    [State('query-two', 'value')]
)
def search(n_clicks, query_one, query_two):
    if n_clicks > 0:
        if not query_one or not query_two:
            return html.Div("Please fill the input fields.", style={'color': 'red'})

        else:
            results = []

            print("q1: ", query_one)
            print("q2: ", query_two)

            score = calculate_similarity(model_bert, tokenizer, params['max_len'], query_one, query_two, device)

            print(score)

            classification = ""

            if score >= 0.75:
                classification = "Entailment"
            elif score < 0.4:
                classification = "Contradiction"
            elif 0.4 <= score < 0.75:
                classification = "Neutral"
            else:
                classification = "Error"

            results.append(html.Div([
                html.H5(f"Calculated similarity score: ", style={'margin-bottom': '10px', 'font-family': 'Arial, sans-serif'}),
                html.P(f"{classification} ({score:.3f})", style={'color': 'black', 'font-family': 'Arial, sans-serif', 'textAlign': 'left'})
            ]))
            
            return html.Div(results, style={
                'background-color': '#f9f9f9',
                'border': '1px solid #e0e0e0',
                'border-radius': '10px',
                'padding': '20px',
                'box-shadow': '0 4px 8px rgba(0, 0, 0, 0.1)',
                'textAlign': 'left',
                'max-width': '50%',
            })

    return html.Div("Enter two sentences to see results.", style={'color': 'gray'})

# Running the app
if __name__ == '__main__':
    app.run_server(debug=True)