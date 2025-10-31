# preprocess.py
import os
import re
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import pickle


nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

class DataPreprocessor:
    def __init__(self, max_sequence_length=50, min_word_frequency=2):
        self.max_sequence_length = max_sequence_length
        self.min_word_frequency = min_word_frequency
        self.tokenizer = None
        self.word_index=0
        self.START_TOKEN = "<start>"
        self.END_TOKEN = "<end>"
        self.UNK_TOKEN = "<unk>"
        self.PAD_TOKEN = "<pad>"
        self.vocab_size = 0
        
    def load_dialogues(self, file_path):
        dialogues = []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    turns = line.split('__eou__')
                    cleaned_turns = [self.clean_text(turn.strip()) for turn in turns if turn.strip()]
                    if len(cleaned_turns) >= 2:
                        dialogues.append(cleaned_turns)
        except FileNotFoundError:
            print(f"Erreur: Fichier {file_path} non trouvé")
            return []
        except Exception as e:
            print(f"Erreur lors de la lecture du fichier {file_path}: {e}")
            return []
        print(f"Chargé {len(dialogues)} dialogues depuis {file_path}")
        return dialogues
    
    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s\.\?\!,]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def create_contextual_pairs(self, dialogues, context_window=2):
        pairs = []
        for dialogue in dialogues:
            for i in range(1, len(dialogue)):
                context_start = max(0, i - context_window)
                context = " | ".join(dialogue[context_start:i])
                response = dialogue[i]
                pairs.append({
                    'context': context,
                    'response': response,
                    'context_turns': dialogue[context_start:i],
                    'response_turn': response
                })
        print(f"Créé {len(pairs)} paires contextuelles")
        return pairs
    
    def build_vocabulary(self, texts):
        all_words = []
        for text in texts:
            tokens = word_tokenize(text)
            all_words.extend(tokens)
        word_freq = Counter(all_words)
        filtered_words = {word for word, count in word_freq.items() if count >= self.min_word_frequency}
        print(f"Vocabulaire: {len(filtered_words)} mots (fréquence >= {self.min_word_frequency})")
        return filtered_words
    
    def create_tokenizer(self, texts):
        vocabulary = self.build_vocabulary(texts)
        self.tokenizer = Tokenizer(filters='', oov_token=self.UNK_TOKEN, lower=False)
        special_tokens = [self.PAD_TOKEN, self.START_TOKEN, self.END_TOKEN, self.UNK_TOKEN]
        for token in special_tokens:
            self.tokenizer.fit_on_texts([token])
        self.tokenizer.fit_on_texts([' '.join(vocabulary)])
        self.vocab_size = len(self.tokenizer.word_index) + 1
        print(f"Taille du vocabulaire final: {self.vocab_size}")
        return self.tokenizer
    
    def preprocess_pairs(self, pairs, is_training=True):
        if is_training and not self.tokenizer:
            all_texts = [pair['context'] for pair in pairs] + [pair['response'] for pair in pairs]
            self.create_tokenizer(all_texts)
        
        input_texts = [pair['context'] for pair in pairs]
        target_texts = [f"{self.START_TOKEN} {pair['response']} {self.END_TOKEN}" for pair in pairs]
        
        input_sequences = self.tokenizer.texts_to_sequences(input_texts)
        target_sequences = self.tokenizer.texts_to_sequences(target_texts)
        
        input_padded = pad_sequences(input_sequences, maxlen=self.max_sequence_length,
                                     padding='post', truncating='post',
                                     value=self.tokenizer.word_index[self.PAD_TOKEN])
        target_padded = pad_sequences(target_sequences, maxlen=self.max_sequence_length,
                                      padding='post', truncating='post',
                                      value=self.tokenizer.word_index[self.PAD_TOKEN])
        print(f"Forme des inputs: {input_padded.shape}")
        print(f"Forme des targets: {target_padded.shape}")
        return input_padded, target_padded
    
    def prepare_datasets(self, base_path):
        train_path = os.path.join(base_path, "train", "dialogues_train.txt")
        val_path = os.path.join(base_path, "validation", "dialogues_validation.txt")
        test_path = os.path.join(base_path, "test", "dialogues_test.txt")
        
        train_dialogues = self.load_dialogues(train_path)
        val_dialogues = self.load_dialogues(val_path)
        test_dialogues = self.load_dialogues(test_path)
        
        train_pairs = self.create_contextual_pairs(train_dialogues)
        val_pairs = self.create_contextual_pairs(val_dialogues)
        test_pairs = self.create_contextual_pairs(test_dialogues)
        
        train_input, train_target = self.preprocess_pairs(train_pairs, is_training=True)
        val_input, val_target = self.preprocess_pairs(val_pairs, is_training=False)
        test_input, test_target = self.preprocess_pairs(test_pairs, is_training=False)
        
        return {
            'train': (train_input, train_target),
            'val': (val_input, val_target),
            'test': (test_input, test_target),
            'tokenizer': self.tokenizer,
            'vocab_size': self.vocab_size,
            'max_sequence_length': self.max_sequence_length,
            'pairs': {'train': train_pairs, 'val': val_pairs, 'test': test_pairs}
        }
    
    def save_preprocessor(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump({
                'tokenizer': self.tokenizer,
                'vocab_size': self.vocab_size,
                'max_sequence_length': self.max_sequence_length,
                'config': {'min_word_frequency': self.min_word_frequency}
            }, f)
        print(f"Prepro sauvegarde : {file_path}")
    
    def load_preprocessor(self, file_path):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            self.tokenizer = data['tokenizer']
            self.vocab_size = data['vocab_size']
            self.max_sequence_length = data['max_sequence_length']
        print(f"Préprocesseur chargé: {file_path}")

def analyze_dataset(pairs, tokenizer, name="Dataset"):
    print(f"\n=== Analyse {name} ===")
    print(f"Nombre de paires: {len(pairs)}")
    context_lengths = [len(pair['context'].split()) for pair in pairs]
    response_lengths = [len(pair['response'].split()) for pair in pairs]
    print(f"Longueur moyenne du contexte: {np.mean(context_lengths):.2f} mots")
    print(f"Longueur moyenne de la réponse: {np.mean(response_lengths):.2f} mots")
    

if __name__ == "__main__":
    import argparse
    BASE_PATH = "D:/sara_projects/projets/chatbot/data" 
    PREPROCESSOR_SAVE = "preprocessor.pkl"
    DATA_SAVE = "dataset_preprocessed.pkl"

    preprocessor = DataPreprocessor(max_sequence_length=50, min_word_frequency=2)


    datasets = preprocessor.prepare_datasets(BASE_PATH)

    preprocessor.save_preprocessor(PREPROCESSOR_SAVE)

    with open(DATA_SAVE, "wb") as f:
        pickle.dump(datasets, f)
