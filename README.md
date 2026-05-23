# Generative Pre-trained Transformer From Scratch

This project implements a decoder-only Generative Pre-trained Transformer from scratch in pure PyTorch. GPT is the backbone of the ChatGPT we all use. It generates output text based on the input text (query).

- Inputs : The query token sequence
- Output : The generated tokens on top of our query.

We can train our GPT on any custom corpus but we use Shakespeare.

## Architecture Overview:

- Tokenizer: The tokenizer's purpose is break down our sentences into tokens. 1 token is roughly 3/4th of a word.
- Embeddings: These are the numerical representations of the tokens in multi dimensional space.
- Positional Encodings: The positional information of the tokens. Helps model understand that "dog bites man" is distinct from "man bites dog" or "man dog bites".
- Multi-Head Attention: We linearly project the queries, keys and values h times with different, learned linear projections to dq, dk, and dv. We perform the attention fucntion parallel on each of these projected versions.
- FeedForward: In transformer specifically two linear projects with a ReLU in between. Input (d_model) -> Linear -> ReLU -> Linear -> Output(d_model)
- Decoder Block: Complete block consisting of Multi-Head Attention -> Add & Norm -> FeedForward -> Add & Norm.
- LM Head: Linear layer that projects from embedding dimension to vocabulary size, converts embeddings back to token probabilities