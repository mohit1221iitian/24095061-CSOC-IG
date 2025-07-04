from google.colab import files
uploaded = files.upload()

import pandas as pd
import numpy as np
import io

filename = list(uploaded.keys())[0]
df = pd.read_csv(io.StringIO(uploaded[filename].decode('utf-8')))
df = df.dropna()
df=df.sample(10000)

for i in df.index:
    df.loc[i, 'hindi'] = '<start> ' + df.loc[i, 'hindi'] + ' <end>'

import tensorflow
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences

hin = Tokenizer(filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n')  # remove < and > from filters
eng=Tokenizer()

print('<start>' in hin.word_index)     # should be True
print(hin.word_index['<start>'])       # should print an integer like 1 or 2

eng.fit_on_texts(df['english'])
hin.fit_on_texts(df['hindi'])

input_seq=eng.texts_to_sequences(df['english'])
output_seq=hin.texts_to_sequences(df['hindi'])

encoder_input=pad_sequences(input_seq,padding='post')
decoder_input=pad_sequences(output_seq,padding='post')

import numpy as np
decoder_target = np.zeros_like(decoder_input)
decoder_target[:, :-1] = decoder_input[:, 1:]

from keras.layers import Dense,LSTM,Embedding,Input,Attention,Concatenate
from keras.models import Model

engunique_words=len(eng.word_index)+1
hinunique_words=len(hin.word_index)+1

enc_emb_layer = Embedding(engunique_words, 256)
dec_emb_layer = Embedding(hinunique_words, 256)

# encoder
enc_input=Input(shape=(None,))
enc_emb = enc_emb_layer(enc_input)
_,state_h,state_c=LSTM(256,return_state=True)(enc_emb)
enc_states=[state_h,state_c]

#decoder
dec_input=Input(shape=(None,))
dec_emb = dec_emb_layer(dec_input)
dec_lstm=LSTM(256,return_sequences=True,return_state=True)
dec_output,_,_=dec_lstm(dec_emb,initial_state=enc_states)
dec_dense=Dense(hinunique_words,activation='softmax')
dec_output=dec_dense(dec_output)

model=Model([enc_input,dec_input],dec_output)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

model.fit([encoder_input, decoder_input], decoder_target[..., np.newaxis],
          batch_size=10, epochs=10, verbose=1)

# Encoder inference model
encoder_model = Model(enc_input, [state_h, state_c])

# Decoder inference model
decoder_state_input_h = Input(shape=(256,))
decoder_state_input_c = Input(shape=(256,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

# ❗ Reuse dec_emb_layer
dec_emb2 = dec_emb_layer(dec_input)

decoder_outputs2, state_h2, state_c2 = dec_lstm(dec_emb2, initial_state=decoder_states_inputs)
decoder_states2 = [state_h2, state_c2]
decoder_outputs2 = dec_dense(decoder_outputs2)

decoder_model = Model(
    [dec_input] + decoder_states_inputs,
    [decoder_outputs2] + decoder_states2
)

def decode_sequence(input_seq):
    # Encode input and get initial states
    states_value = encoder_model.predict(input_seq)

    # Start with the <start> token
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = hin.word_index['<start>']

    stop_condition = False
    decoded_sentence = []

    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # Get the predicted token index
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = hin.index_word.get(sampled_token_index, '')

        if (sampled_word == '<end>' or len(decoded_sentence) > 30):
            stop_condition = True
        else:
            decoded_sentence.append(sampled_word)

        # Update the target sequence and states
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        states_value = [h, c]

    return ' '.join(decoded_sentence)

def translate(text):
    seq = eng.texts_to_sequences([text])
    seq = pad_sequences(seq, maxlen=encoder_input.shape[1], padding='post')  # same as training
    return decode_sequence(seq)

print(translate("why"))

engunique_words=len(eng.word_index)+1
hinunique_words=len(hin.word_index)+1

# encoder
enc_input=Input(shape=(None,))
enc_emb=Embedding(engunique_words,256)(enc_input)
enc_output,state_h,state_c=LSTM(256,return_state=True,return_sequences=True
)(enc_emb)
enc_states=[state_h,state_c]

#decoder
dec_input=Input(shape=(None,))
dec_emb=Embedding(hinunique_words,256)(dec_input)
dec_lstm=LSTM(256,return_sequences=True,return_state=True)
dec_lstm_output,_,_=dec_lstm(dec_emb,initial_state=enc_states)

attention = Attention()([dec_lstm_output, enc_output])
concat = Concatenate(axis=-1)([dec_lstm_output, attention])
outputs = Dense(hinunique_words, activation='softmax')(concat)

model1=Model([enc_input,dec_input],outputs)
model1.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

model1.fit([encoder_input, decoder_input], decoder_target[..., np.newaxis],
          batch_size=10, epochs=10, verbose=1)

from nltk.translate.bleu_score import sentence_bleu, corpus_bleu

def decode_sequence(input_seq):

    states_value = model.predict(input_seq)

    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = hin.word_index['<start>']

    stop_condition = False
    decoded_sentence = []

    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = hindi_tokenizer.index_word.get(sampled_token_index, '')

        if sampled_word == '<end>' or len(decoded_sentence) > max_decoder_seq_length:
            stop_condition = True
        else:
            decoded_sentence.append(sampled_word)

            # Update target_seq and states
            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = sampled_token_index
            states_value = [h, c]

    return ' '.join(decoded_sentence)
