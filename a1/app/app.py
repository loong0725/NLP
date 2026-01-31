from dash import Dash, Input, Output, dcc, html
import numpy as np
import pickle
from heapq import nlargest
import os

# Dash app setup
app = Dash(__name__)


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")

with open(os.path.join(MODEL_DIR, "embed_skipgram.pkl"), "rb") as f:
    skip_gram_model = pickle.load(f)

with open(os.path.join(MODEL_DIR, "embed_negative_samp.pkl"), "rb") as f:
    negative_sampling_model = pickle.load(f)

with open(os.path.join(MODEL_DIR, "embed_glove.pkl"), "rb") as f:
    glove_model = pickle.load(f)



# Ensure models are compatible with embeddings
embedding_dicts = {
    "Skip-gram": skip_gram_model,
    "Negative Sampling": negative_sampling_model,
    "GloVe": glove_model,
}

# Cosine similarity function
def cosine_similarity(A, B):
    dot_product = np.dot(A, B)
    norm_a = np.linalg.norm(A)
    norm_b = np.linalg.norm(B)
    similarity = dot_product / (norm_a * norm_b)
    return similarity

# Find top N similar words using cosine similarity
def find_next_10_cosine_words_for_word(target_word, embeddings, top_n=10):
    if target_word not in embeddings:
        return ["Word not in Corpus"]

    target_vector = embeddings[target_word]
    cosine_similarities = [
        (word, cosine_similarity(target_vector, embeddings[word]))
        for word in embeddings.keys()
    ]
    top_n_words = nlargest(top_n + 1, cosine_similarities, key=lambda x: x[1])

    # Exclude the target word itself
    top_n_words = [word for word, _ in top_n_words if word != target_word]

    return top_n_words[:top_n]

# Dash layout
app.layout = html.Div(
    [
        html.H1(
            "Word Context Prediction",
            style={
                "textAlign": "center",
                "margin": "0 auto",
                "padding": "20px",
            }),
        html.Div(
            dcc.Dropdown(
                id="embedding_selector",
                options=[{"label": key, "value": key} for key in embedding_dicts.keys()],
                value="Skip-gram",
                placeholder="Select an embedding model",
                style={"width": "60%", "margin": "0 auto", "padding": "5px"},
            ),
            style={
                "display": "flex",
                "justifyContent": "center",
                "alignItems": "center",
                "marginBottom": "20px",
            },
        ),
        html.Div(
            dcc.Input(
                id="search_query",
                type="text",
                placeholder="Enter your word",
                style={"width": "30%", "padding": "5px"},
            ),
            style={
                "display": "flex",
                "justifyContent": "center",
                "marginBottom": "20px",
            },
        ),
        html.Button(
            "Predict",
            id="predict_button",
            n_clicks=0,
            style={
                "margin": "20px auto",
                "display": "block",
                "backgroundColor": "#007BFF",
                "color": "white",
                "border": "none",
                "borderRadius": "5px",
                "padding": "10px 20px",
                "cursor": "pointer",
                "fontSize": "16px",
            },
        ),
        # Centering results and data store
        html.Div(
            [
                html.Div(id="results", style={"textAlign": "center", "marginTop": "20px"}),
            ],
            style={
                "display": "flex",
                "flexDirection": "column",
                "alignItems": "center",
                "marginTop": "20px",
            },
        ),
    ]
)


# Callback to handle prediction
@app.callback(
    Output("results", "children"),
    [Input("predict_button", "n_clicks")],
    [Input("embedding_selector", "value"), Input("search_query", "value")],
)
def predict_context(n_clicks, embedding_name, query):
    if n_clicks == 0 or not query:
        return None  # Don't show results until button is clicked

    # Get the embedding dictionary
    embedding_dict = embedding_dicts.get(embedding_name, {})

    # Find similar words
    similar_words = find_next_10_cosine_words_for_word(query, embedding_dict, top_n=10)

    # Format the resultsS
    if similar_words == ["Word not in Corpus"]:
        return "Word not found in the embedding corpus."

    results = html.Div(
        [html.Div(f"{i + 1}. {word}") for i, word in enumerate(similar_words)]
    )

    return results


# Run the app
if __name__ == "__main__":
    app.run(debug=True, port=8050)
