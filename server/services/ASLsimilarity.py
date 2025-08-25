import pandas as pd
import numpy as np
import re
import torch
from sentence_transformers import SentenceTransformer, util
from nltk.stem import WordNetLemmatizer
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig
from sklearn.metrics.pairwise import cosine_similarity
import warnings
from transformers import logging
from openai import OpenAI

# Suppress warnings about uninitialized weights
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub.file_download")
warnings.filterwarnings("ignore", message="Some weights of the model checkpoint")
logging.set_verbosity_error()

client = OpenAI(api_key="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") #Lab-Tips

name_df = pd.read_csv('data/name.csv')
name_set = set(name_df['Name'].str.lower())

def gpt_run(prompt_system: str, prompt_user: str):
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": prompt_system},
            {"role": "user", "content": prompt_user}
        ]
    )
    print(f"GPT response: {completion.choices[0].message}")
    return completion.choices[0].message.content

class ASLSimilarity:
    def __init__(self, sign_df, definition_embeddings_df, gloss_embeddings_df, sentence_embeddings_df):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # # SentenceTransformer for initial definition-level similarity
        # self.model_paraphrase = SentenceTransformer('paraphrase-MiniLM-L6-v2').to(self.device)
        
        # RoBERTa for final candidate selection based on word-level embedding
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        config = RobertaConfig.from_pretrained('roberta-base', output_hidden_states=True, add_pooling_layer=False)
        self.roberta_model = RobertaModel.from_pretrained('roberta-base', config=config, ignore_mismatched_sizes=True).to(self.device)
        self.roberta_model.eval()

        # self.sign_df = sign_df
        # only keep the entries with non-empty definitions
        self.sign_df = sign_df[sign_df['Definition'].notnull()]
        self.definition_embeddings_df = definition_embeddings_df[sign_df['Definition'].notnull()]
        self.gloss_embeddings_df = gloss_embeddings_df[sign_df['Definition'].notnull()]
        self.sentence_embeddings_df = sentence_embeddings_df[sign_df['Definition'].notnull()]
        # self.top_num = top_num

        self._prepare_data()
        
        self.definition_embeddings = self._load_embeddings(self.definition_embeddings_df)
        self.gloss_embeddings = self._load_embeddings(self.gloss_embeddings_df)
        self.sentence_embeddings = self._load_embeddings(self.sentence_embeddings_df)
        
        self.lemmatizer = WordNetLemmatizer()
        
    def _prepare_data(self):
        self.definitions = self.sign_df['Definition'].tolist()
        self.entry_ids = self.sign_df['Entry'].tolist()
        self.glosses = self.sign_df['Gloss'].tolist()
        self.sentences = self.sign_df['Sentence'].tolist()  # newly added column

    def _load_embeddings(self, embeddings_df):
        return torch.tensor(embeddings_df.values, dtype=torch.float32)
    
    # def _get_embeddings(self, texts):
    #     embeddings = self.model_paraphrase.encode(texts, convert_to_tensor=True)
    #     return embeddings.to(self.device)

    def _lemmatize_word(self, word):
        word_lower = word.lower()
        lemma_verb = self.lemmatizer.lemmatize(word_lower, pos='v')
        if lemma_verb == word_lower:
            lemma_noun = self.lemmatizer.lemmatize(word_lower, pos='n')
            return lemma_noun
        else:
            return lemma_verb
    
    def _get_word_embedding_roberta(self, sentence, target_word):
        """
        Extracts the embedding of a target_word as it appears in a sentence using RoBERTa.
        Averages over subword tokens if the word is split into multiple tokens.
        Returns a numpy array of shape (hidden_dim,).
        If word not found, returns None.
        """
        inputs = self.tokenizer(sentence, return_tensors='pt').to(self.device)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        
        with torch.no_grad():
            outputs = self.roberta_model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        last_hidden_state = outputs.last_hidden_state[0]
        
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        
        # Clean tokens for matching
        def clean_token(tok):
            return tok.lstrip('Ġ').lower()
        
        target_word_lower = target_word.lower()
        candidate_indices = []
        # First attempt: direct token match
        for i, tok in enumerate(tokens):
            if tok in ['<s>', '</s>']:
                continue
            if clean_token(tok) == target_word_lower:
                candidate_indices = [i]
                break
        
        # If no direct match, try subword concatenation
        if not candidate_indices:
            for start_idx in range(len(tokens)):
                if tokens[start_idx] in ['<s>', '</s>']:
                    continue
                concat_word = clean_token(tokens[start_idx])
                end_idx = start_idx
                while len(concat_word) < len(target_word_lower) and end_idx + 1 < len(tokens):
                    end_idx += 1
                    if tokens[end_idx] in ['<s>', '</s>']:
                        break
                    concat_word += clean_token(tokens[end_idx])
                if concat_word == target_word_lower:
                    candidate_indices = list(range(start_idx, end_idx + 1))
                    break
        
        if not candidate_indices:
            return None
        
        word_emb = last_hidden_state[candidate_indices].mean(dim=0).cpu().numpy()
        return word_emb

    def find_top_n_entries_gloss_sent(self, sentence, gloss_sentence, target_word, n=1):
        """
        special case:name
        special_case:hard code

        for polysemy, use gpt
        
        # alpha = 0.5
        # beta = 0.5
        candidates = []
        # exact_match = False
        
        for i, gloss in enumerate(self.glosses):
            if definition is not empty:
                if target_word == gloss:
                    candidates.append(i)
                    exact_match = True
                elif target_word in definition:
                    candidates.append(i)
        if len(candidates) == 0:
            if target_word in ur:
                return
            else:
                candidates = [i for i in range(len(self.glosses)) if definition is not empty]
        for cand in candidates:
            Compute similarity of candidates's in-sent embedding with target_word's in-sent embedding
            # if exact_match and len(candidates) > 1:
            #     weighted_similarity = alpha * in_sent_similarity + beta * definition_similarity
        """
        
        # Special cases
        if target_word.lower() in name_set:
            print(f"Target word is a person name: {target_word}")
            # return list of letters in the name, capitalized first letter
            target_word = target_word[0].upper() + target_word[1:].lower()
            return [("name_fingerspell", "name", target_word, 1)]

        target_word_lower = target_word.lower()
        if target_word_lower in ['who', 'when', 'where', 'why', 'which']:
            gloss_idx = self.entry_ids.index(f'L_{target_word_lower}')
            return [(self.entry_ids[gloss_idx], self.definitions[gloss_idx], self.glosses[gloss_idx], 1)]
        elif target_word_lower == 'how':
            gloss_idx = self.entry_ids.index(f'L_{target_word_lower}_2')
            return [(self.entry_ids[gloss_idx], self.definitions[gloss_idx], self.glosses[gloss_idx], 1)]
        # elif target_word_lower in ['how-many','how-many'.title()]:
        #     gloss_idx = self.entry_ids.index('HOW-MANY')
        #     return [(self.entry_ids[gloss_idx], self.definitions[gloss_idx], self.glosses[gloss_idx], 1)]
        # elif target_word_lower == 'how-about':
        #     gloss_idx = self.entry_ids.index('HOW-HOWpinky')
        #     return [(self.entry_ids[gloss_idx], self.definitions[gloss_idx], self.glosses[gloss_idx], 1)]
        elif target_word_lower == 'what':
            gloss_idx = self.entry_ids.index(f'L_{target_word_lower}_1')
            return [(self.entry_ids[gloss_idx], self.definitions[gloss_idx], self.glosses[gloss_idx], 1)]
        elif target_word_lower in ['me','i']:
            gloss_idx = self.entry_ids.index(f'L_me')
            return [(self.entry_ids[gloss_idx], self.definitions[gloss_idx], self.glosses[gloss_idx], 1)]
        elif target_word_lower in ['us','we']:
            gloss_idx = self.entry_ids.index(f'L_we')
            return [(self.entry_ids[gloss_idx], self.definitions[gloss_idx], self.glosses[gloss_idx], 1)]
        elif target_word_lower in ['she','he','him','her']:
            gloss_idx = self.entry_ids.index(f'L_he')
            return [(self.entry_ids[gloss_idx], self.definitions[gloss_idx], self.glosses[gloss_idx], 1)]
        elif target_word_lower == ['their','them']:
            gloss_idx = self.entry_ids.index(f'L_they_1')
            return [(self.entry_ids[gloss_idx], self.definitions[gloss_idx], self.glosses[gloss_idx], 1)]
        elif target_word_lower == 'hers':
            gloss_idx = self.entry_ids.index(f'L_his')
            return [(self.entry_ids[gloss_idx], self.definitions[gloss_idx], self.glosses[gloss_idx], 1)]
        elif target_word_lower == 'ours':
            gloss_idx = self.entry_ids.index(f'L_our')
            return [(self.entry_ids[gloss_idx], self.definitions[gloss_idx], self.glosses[gloss_idx], 1)]
        elif target_word_lower in ['it','this']:
            gloss_idx = self.entry_ids.index(f'L_this_it')
            return [(self.entry_ids[gloss_idx], self.definitions[gloss_idx], self.glosses[gloss_idx], 1)]
        elif target_word_lower == 'cozy':
            gloss_idx = self.entry_ids.index(f'L_comfortable')
            return [(self.entry_ids[gloss_idx], self.definitions[gloss_idx], self.glosses[gloss_idx], 1)]
        elif target_word_lower == 'drive':
            gloss_idx = self.entry_ids.index(f'L_car')
            return [(self.entry_ids[gloss_idx], self.definitions[gloss_idx], self.glosses[gloss_idx], 1)]
        elif target_word_lower in ['leg','legs','tentacle','tentacles']:
            gloss_idx = self.entry_ids.index(f'L_pants')
            return [(self.entry_ids[gloss_idx], self.definitions[gloss_idx], self.glosses[gloss_idx], 1)]
        elif target_word_lower in ['previous','previously']:
            gloss_idx = self.entry_ids.index(f'L_before')
            return [(self.entry_ids[gloss_idx], self.definitions[gloss_idx], self.glosses[gloss_idx], 1)]
        
        elif target_word_lower in ['look_at','look-at']:
            target_word_lower = 'look'

        target_word_lemma = self._lemmatize_word(target_word_lower)
        target_word_lemma = re.sub(r'[^\w\s]', '', target_word_lemma)

        # special case: retrieve the entries and definitions for the target word, use gpt to choose the best one
        # example prompt: "The bat flew silently across the moonlit sky. In this sentence, whether the bat means (1) hit, hit stick, knock, strike, swing, or (2) flying mammal? Return only the index of the right answer"
        if target_word_lemma in ['bat','can','call','dark','fall','frown','glass','mean','model','order','patient','right','type','like','tear']:
            def_list = []
            for i, gloss in enumerate(self.glosses):
                if target_word_lemma in gloss:
                    def_list.append((i, self.definitions[i]))
                if target_word_lemma in ['glass','like'] and target_word_lemma in self.definitions[i]:
                    def_list.append((i, self.definitions[i]))
            prompt_system = ""
            prompt_user = f"{sentence}. In this sentence, whether the {target_word_lemma} means "
            for i, (idx, definition) in enumerate(def_list):
                prompt_user += f"({i+1}) {definition}, "
            prompt_user += "? Return only the index of the right answer"
            response = gpt_run(prompt_system, prompt_user).strip()
            print(f"GPT response: {response}")
            
            # Extract the first digit from the response (if any)
            match_digits = re.findall(r"\d+", response)
            if not match_digits:
                print(f"Invalid response (no digit found): {response}")
            else:
                chosen_index = int(match_digits[0])
                if 1 <= chosen_index <= len(def_list):
                    gloss_idx = def_list[chosen_index - 1][0]
                    return [(self.entry_ids[gloss_idx],self.definitions[gloss_idx],self.glosses[gloss_idx],1)]
                else:
                    print(f"Invalid response (index out of range): {response}")
            
        # alpha = 0.5
        # beta = 0.5
        
        candidates = []
        def_candidates = []
        # exact_match = False
        for i, gloss in enumerate(self.glosses):
            gloss_lower = gloss.lower().strip()
            definition_lower = self.definitions[i].lower().strip()
            # Split definition by comma
            def_terms = [d.strip() for d in definition_lower.split(',')]
            
            if gloss_lower in [target_word_lemma, target_word_lower]:
                # Candidate matches exact conditions
                candidates.append(i)
                # exact_match = True
            elif target_word_lemma in def_terms or target_word_lower in def_terms:
                # Candidate matches target word in definition
                def_candidates.append(i)
        print(f"candidates: {[self.entry_ids[i] for i in candidates]}")
        if len(candidates) == 0:
            candidates = def_candidates
        
        if len(candidates) == 0:
            # # check if target_word is in ur_df gloss
            # if target_word_lemma in self.ur_df['Gloss'].values:
            #     print(f"Target word '{target_word_lemma}' found in UR glosses.")
            #     ur_idx = self.ur_df[self.ur_df['Gloss'] == target_word_lemma].index[0]
            #     return [(self.ur_df.loc[ur_idx, 'Entry'], self.ur_df.loc[ur_idx, 'Definition'], target_word_lemma, 1)]
            
            # # no candidates match exactly, consider all entries
            # else:
            print(f"No exact matches found for target word '{target_word_lower}'. Considering all entries.")
            candidates = [i for i in range(len(self.glosses))]

        # Use RoBERTa to pick the best one based on word-in-context similarity.
        # Get word embedding from input sentence for target_word
        # lemmatize, remove punctuation, and lowercase gloss_sentence
        sentence = re.sub(r'[^\w\s]', '', sentence.lower())
        sentence = ' '.join([self._lemmatize_word(word) for word in sentence.split()])
        input_word_emb = self._get_word_embedding_roberta(sentence.lower(), target_word_lower)
        if input_word_emb is None:
            # If we cannot find the target word in the input sentence, use gloss sentence
            print(f"Target word '{target_word_lower}' not found in input sentence. Using gloss sentence.")
            input_word_emb = self._get_word_embedding_roberta(gloss_sentence.lower(), target_word_lower)
            if input_word_emb is None:
                print(f"Target word '{target_word_lower}' not found in gloss sentence. Using the first candidate.")
                best_idx = candidates[0]
                return [(self.entry_ids[best_idx], self.definitions[best_idx], self.glosses[best_idx], 1)]
        
        best_entry = None
        best_score = -1
        
        for i in candidates:
            # Get gloss embedding from the candidate entry's sentence
            candidate_emb = self.sentence_embeddings[i]
            # # if zero vector, print i
            # if torch.all(candidate_emb == 0):
            #     print(f"Zero vector for candidate {i}")
            
            if candidate_emb is not None:
                # if exact_match and len(candidates) > 1:
                # Compute similarity of candidate's in-sentence embedding with target_word's in-sentence embedding
                in_sent_similarity = cosine_similarity(candidate_emb.reshape(1, -1), input_word_emb.reshape(1, -1))[0][0]
                # # Compute similarity of candidate's definition embedding with target_word's in-sentence embedding
                # definition_similarity = cosine_similarity(self.definition_embeddings[i].reshape(1, -1), input_word_emb.reshape(1, -1))[0][0]
                # weighted_similarity = alpha * in_sent_similarity + beta * definition_similarity
                # print(f"Candidate {self.entry_ids[i]}: in_sent:{in_sent_similarity}, def:{definition_similarity}, weighted:{weighted_similarity}")
                if in_sent_similarity > best_score:
                    best_score = in_sent_similarity
                    best_entry = i
                # else:
                #     # Compute cosine similarity of candidate word in sentence embedding with target word in sentence embedding
                #     sim_score = cosine_similarity(input_word_emb.reshape(1, -1), candidate_emb.reshape(1, -1))[0][0]
                #     # print(f"Candidate {self.entry_ids[i]}: {sim_score}")
                #     if sim_score > best_score:
                #         best_score = sim_score
                #         best_entry = i
            else:
                print(f"Could not get embedding for candidate {i}")
                continue
        
        return [(self.entry_ids[best_entry], self.definitions[best_entry], self.glosses[best_entry], best_score)]




# # Load data
# sign_df = pd.read_csv('data/signdf.csv')
# definition_embeddings_df = pd.read_csv('data/definition_embeddings.csv')
# gloss_embeddings_df = pd.read_csv('data/gloss_embeddings.csv')
# sentence_embeddings_df = pd.read_csv('data/sentence_embeddings.csv')

# # set the index to only consider rows 1-2715 in the dataframe
# sign_df = sign_df.iloc[:2715]
# definition_embeddings_df = definition_embeddings_df.iloc[:2715]
# gloss_embeddings_df = gloss_embeddings_df.iloc[:2715]
# sentence_embeddings_df = sentence_embeddings_df.iloc[:2715]



# asl_similarity = ASLSimilarity(sign_df, definition_embeddings_df, gloss_embeddings_df, sentence_embeddings_df)

# sentence = "Where is the restaurant?"
# gloss_sentences = "RESTAURANT WHERE"
# target_word = "restaurant"

# entry_id, definition, gloss, score= asl_similarity.find_top_n_entries_gloss_sent(sentence, gloss_sentences, target_word)[0]
# ret = {
#             'sentence': sentence,
#             'gloss_sent': gloss_sentences,
#             'target_word': target_word,
#             # 'target_entryID': target_entryID,
#             # 'entry_id': ', '.join(entry_list),
#             'entry_id': entry_id,
#             'definition': definition,
#             'gloss': gloss,
#             'score': score
#         }
# print(ret)