import torch
import torchtext
import torch.nn as nn
import dill
import numpy as np
from torch.autograd import Variable
from torch.nn import functional as F

class ClassifierModel(nn.Module):
    def __init__(self, batch_size, output_size, hidden_size, vocab_size, embedding_length, weights):
        super(ClassifierModel, self).__init__()
        """
        output_size : 2 = (pos, neg)
        """
        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding_length = embedding_length

        self.word_embeddings = nn.Embedding(vocab_size, embedding_length)  # Initiale the look-up table.
        self.word_embeddings.weight = nn.Parameter(weights, requires_grad=False) # Assign pre-trained GloVe word embedding.
        self.lstm = nn.LSTM(embedding_length, hidden_size)
        self.label = nn.Linear(hidden_size, output_size)

    def forward(self, input_sentence, batch_size=None):
        """ 
        final_output.shape = (batch_size, output_size)
        """
        input = self.word_embeddings(input_sentence) # embedded input of shape = (batch_size, num_sequences,  embedding_length)
        input = input.permute(1, 0, 2) # input.size() = (num_sequences, batch_size, embedding_length)
        if batch_size is None:
            h_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size)) # Initial hidden state of the LSTM
            c_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size)) # Initial cell state of the LSTM
        else:
            h_0 = Variable(torch.zeros(1, batch_size, self.hidden_size))
            c_0 = Variable(torch.zeros(1, batch_size, self.hidden_size))
        output, (final_hidden_state, final_cell_state) = self.lstm(input, (h_0, c_0))
        final_output = self.label(final_hidden_state[-1]) # final_hidden_state.size() = (1, batch_size, hidden_size) & final_output.size() = (batch_size, output_size)

        return final_output

batch_size = 32
output_size = 11
hidden_size = 256
word_embeddings = torch.load('word_embeddings.pt')
vocab_size = 439213
embedding_length = 100
model = ClassifierModel(batch_size, output_size, hidden_size, vocab_size, embedding_length, word_embeddings)
model.load_state_dict(torch.load('saved_weights.pt',map_location=torch.device('cpu')))
with open("TEXT.Field","rb")as f:
    TEXT=dill.load(f)
userinput = "Hello world"
text = userinput.lower()
test_sent = TEXT.preprocess(text)
test_sent = [[TEXT.vocab.stoi[x] for x in test_sent]]
test_sent = np.asarray(test_sent)
test_sent = torch.LongTensor(test_sent)
test_tensor = Variable(test_sent)
# test_tensor = test_tensor.cuda()
model.eval()
output = model(test_tensor, 1)
out = F.softmax(output, 1)
print(torch.argmax(out[0]).item())