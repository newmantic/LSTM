# LSTM

Overview
Long Short-Term Memory (LSTM) is a type of Recurrent Neural Network (RNN) architecture designed to capture long-term dependencies in sequential data. It addresses the vanishing gradient problem in traditional RNNs, making it more effective at learning patterns over longer sequences.



Cell State: The cell state c_t is a vector that carries information across different time steps in the sequence. It acts as a kind of memory for the LSTM.

Hidden State: The hidden state h_t is another vector that is used to compute the output at each time step. It is also passed to the next time step.

Gates: LSTM uses three gates to control the flow of information:
Forget Gate f_t: Determines how much of the previous cell state c_(t-1) should be forgotten.
Input Gate i_t: Controls how much of the new information should be added to the cell state.
Output Gate o_t: Decides how much of the cell state should be output to the hidden state.



Input Sequence: Let x_1, x_2, ..., x_T be the input sequence, where each x_t is a vector representing the input at time step t. The sequence length is T.

Forget Gate: The forget gate f_t decides what fraction of the previous cell state c_(t-1) should be carried forward. It is computed as:
f_t = sigmoid(W_f * [h_(t-1), x_t] + b_f)
where:
W_f is the weight matrix for the forget gate.
b_f is the bias for the forget gate.
h_(t-1) is the hidden state from the previous time step.
x_t is the current input.

Input Gate: The input gate i_t controls how much of the new information (candidate cell state c~_t) should be added to the cell state c_t. It is computed as:
i_t = sigmoid(W_i * [h_(t-1), x_t] + b_i)
The candidate cell state c~_t is computed as:
c~_t = tanh(W_c * [h_(t-1), x_t] + b_c)
where:
W_i and W_c are the weight matrices for the input gate and candidate cell state, respectively.
b_i and b_c are the biases for the input gate and candidate cell state, respectively.

Cell State Update: The cell state c_t is updated using the forget gate f_t, the previous cell state c_(t-1), the input gate i_t, and the candidate cell state c~_t:
c_t = f_t * c_(t-1) + i_t * c~_t

Output Gate: The output gate o_t controls how much of the cell state c_t should be output to the hidden state h_t. It is computed as:
o_t = sigmoid(W_o * [h_(t-1), x_t] + b_o)
where:
W_o is the weight matrix for the output gate.
b_o is the bias for the output gate.

Hidden State Update: The hidden state h_t is updated using the output gate o_t and the current cell state c_t:
h_t = o_t * tanh(c_t)


Initialization:
Initialize the cell state c_0 and the hidden state h_0 to zero vectors.

Forward Pass:
For each time step t:
Compute the forget gate f_t.
Compute the input gate i_t and candidate cell state c~_t.
Update the cell state c_t.
Compute the output gate o_t.
Update the hidden state h_t.

Output:
The final hidden state h_T or the entire sequence of hidden states can be used as the output of the LSTM.
