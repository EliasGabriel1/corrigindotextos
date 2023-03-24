import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint

# Carrega o corpus
nltk.download('punkt')
corpus = nltk.corpus.gutenberg.raw('austen-emma.txt')

# Prepara os dados
tokens = word_tokenize(corpus.lower())
unique_tokens = sorted(list(set(tokens)))
token_to_index = dict((c, i) for i, c in enumerate(unique_tokens))
index_to_token = dict((i, c) for i, c in enumerate(unique_tokens))
vocab_size = len(unique_tokens)
seq_length = 25

# Cria as sequências de entrada e saída
input_seqs = []
output_seqs = []
for i in range(0, len(tokens) - seq_length):
    input_seq = tokens[i:i + seq_length]
    output_seq = tokens[i + seq_length]
    input_seqs.append([token_to_index[token] for token in input_seq])
    output_seqs.append(token_to_index[output_seq])

# Transforma as sequências em arrays numpy
X = np.array(input_seqs)
Y = to_categorical(output_seqs)

# Define o modelo da rede neural
model = Sequential()
model.add(LSTM(256, input_shape=(seq_length, 1)))
model.add(Dropout(0.2))
model.add(Dense(Y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Treina o modelo
filepath = "weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
model.fit(X, Y, epochs=20, batch_size=128, callbacks=callbacks_list)

# Gera a próxima palavra da frase
def generate_word(model, input_seq):
    x = np.reshape(input_seq, (1, len(input_seq), 1))
    x = x / float(vocab_size)
    pred = model.predict(x, verbose=0)
    index = np.argmax(pred)
    return index_to_token[index]

# Gera uma frase corrigida
def correct_sentence(model, sentence):
    input_seq = [token_to_index[token] for token in word_tokenize(sentence.lower())]
    while len(input_seq) < seq_length:
        input_seq.insert(0, 0)

        # passando a palavra pra letrar minusculas
    corrected_sentence = sentence.lower()
    for i in range(len(sentence), len(input_seq)):
        next_word = generate_word(model, input_seq[i - seq_length:i])
        corrected_sentence += ' ' + next_word
        input_seq[i] = token_to_index[next_word]
    return corrected_sentence.capitalize()

# Testa o corretor de frases
sentence = "She is a wery goood persen."
corrected_sentence = correct_sentence(model, sentence)
print("Original sentence: ", sentence)
print("Corrected sentence: ", corrected_sentence)