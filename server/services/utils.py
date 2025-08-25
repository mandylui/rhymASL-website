#%%
from text_to_gloss_translation import generate_asl_gloss_wo_nms
from ASLsimilarity import ASLSimilarity,gpt_run
import pandas as pd
import os
import pandas as pd
import numpy as np
import time

#%%
# find the video entry for the target word
# Load data
sign_df = pd.read_csv('data/signdf.csv')
definition_embeddings_df = pd.read_csv('data/definition_embeddings.csv')
gloss_embeddings_df = pd.read_csv('data/gloss_embeddings.csv')
sentence_embeddings_df = pd.read_csv('data/sentence_embeddings.csv')
# set the index to only consider rows 0-2715 in the dataframe
sign_df = sign_df.iloc[:2715]
definition_embeddings_df = definition_embeddings_df.iloc[:2715]
gloss_embeddings_df = gloss_embeddings_df.iloc[:2715]
sentence_embeddings_df = sentence_embeddings_df.iloc[:2715]
asl_similarity = ASLSimilarity(sign_df, definition_embeddings_df, gloss_embeddings_df, sentence_embeddings_df)

# read signdata, entry_code_check, and phonology
signdata = pd.read_csv('osfstorage-archive/Data_Files/signdata.csv',encoding="ISO-8859-1")
entry_code_check = pd.read_csv('osfstorage-archive/Data_Files/entry_code_check.csv')
# loop A to K
phono_name_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K']
# create a list of phono dataframes names
phono_A = pd.read_csv('osfstorage-archive/Data_Files/Phonology/split_by_prefix/A.csv')
phono_B = pd.read_csv('osfstorage-archive/Data_Files/Phonology/split_by_prefix/B.csv')
phono_C = pd.read_csv('osfstorage-archive/Data_Files/Phonology/split_by_prefix/C.csv')
phono_D = pd.read_csv('osfstorage-archive/Data_Files/Phonology/split_by_prefix/D.csv')
phono_E = pd.read_csv('osfstorage-archive/Data_Files/Phonology/split_by_prefix/E.csv')
phono_F = pd.read_csv('osfstorage-archive/Data_Files/Phonology/split_by_prefix/F.csv')
phono_G = pd.read_csv('osfstorage-archive/Data_Files/Phonology/split_by_prefix/G.csv')
phono_H = pd.read_csv('osfstorage-archive/Data_Files/Phonology/split_by_prefix/H.csv')
phono_J = pd.read_csv('osfstorage-archive/Data_Files/Phonology/split_by_prefix/J.csv')
phono_K = pd.read_csv('osfstorage-archive/Data_Files/Phonology/split_by_prefix/K.csv')
phono_list = [phono_A, phono_B, phono_C, phono_D, phono_E, phono_F, phono_G, phono_H, phono_J, phono_K]

entry_code_check = pd.read_csv('osfstorage-archive/Data_Files/entry_code_check.csv')

#%%
# This function retrieves the gloss from a given sentence.
def get_gloss_from_sentence(sentence,related=dict()):
    """
    This function takes a sentence as input, generates the ASL gloss for each word in the sentence,
    and retrieves the top matching entries from the ASL similarity model.
    :param sentence: str, input sentence to be processed
    :param related: list, list of related {gloss:entry_id} to be considered
    :return: tuple, (list of ASL glosses, list of sign video entry IDs)
    """
    if related is None:
        related = {}
    else:
        for r in related:
            if r.startswith('L_'):
                related[related.index(r)] = r[2:]  # Remove 'L_' prefix if it exists
    start_time = time.time()
    asl = generate_asl_gloss_wo_nms(sentence)
    ## TODO: remove after demonstration
    if sentence == "if you give a pig a party":
        asl = "if you give pig party"
    elif sentence == "brown bear, brown bear, what do you see?":
        asl = "brown bear brown bear you see what"
    elif sentence == "The dirty pig finds a sweet candy.":
        asl = "pig dirty find candy sweet"
    elif sentence == "The dirty pig finds a sweet apple.":
        asl = "pig dirty find apple sweet"
    # lowercase the gloss
    asl = asl.lower()
    print(f"Generate ASL gloss took {time.time() - start_time:.2f} seconds")
    asl_list = asl.split(" ")
    gloss_sentences = ' '.join(asl_list)
    entry_list = []
    lemma_list = []
    for target_word in asl_list:
        word_video, definition, gloss, score = asl_similarity.find_top_n_entries_gloss_sent(sentence, gloss_sentences, target_word)[0]
        word_entry = word_video[2:] if word_video.startswith('L_') else word_video
        print(f"Word: {target_word}, Entry: {word_entry}, Video: {word_video}, Definition: {definition}, Gloss: {gloss}, Score: {score}")
        if target_word in related.keys():
            entry = related[target_word]
            entry_list.append(entry)
            lemma_list.append(target_word)
            continue
        elif gloss in related.keys():
            entry = related[gloss]
            entry_list.append(entry)
            lemma_list.append(gloss)
            continue
        entry_list.append(word_entry)
        lemma_list.append(gloss)
    print(f"Total time for gloss generation: {time.time() - start_time:.2f} seconds")
    print(f"utils-get_gloss_from_sentence: ASL Glosses: {asl_list}, Entry IDs: {entry_list}, Lemma IDs: {lemma_list}")
    return asl_list, entry_list, lemma_list

#%%
def get_sign_video(entry_id):
    """
    This function retrieves the video path for a given sign video entry ID.
    :param entry_id: str, the entry ID of the sign video
    :return: str, path to the sign video file
    """
    if entry_id.startswith('L_'):
        entry_id = entry_id[2:]
    video_path = f"ASL_LEX_MP4/L_{entry_id}.mp4"
    try:
        # find sign duration and clip duration in signdata
        clip_duration = signdata[signdata['EntryID'] == entry_id]['ClipDuration(ms)'].values[0]
        lemma_id = signdata[signdata['EntryID'] == entry_id]['LemmaID'].values[0]
        return video_path, clip_duration,lemma_id
    except Exception as e:
        print(f"Error retrieving video for entry ID '{entry_id}': {e}")
        return None, None, None

def get_sign_videos(entry_ids):
    video_path_list = []
    clip_duration_list = []
    lemma_id_list = []
    for entry_id in entry_ids:
        video_path, clip_duration, lemma_id = get_sign_video(entry_id)
        video_path_list.append(video_path)
        clip_duration_list.append(clip_duration)
        lemma_id_list.append(lemma_id)
    return video_path_list, clip_duration_list, lemma_id_list


#%%
def find_sign_by_handshape(entryID,handshape,sort_flag=['Iconicity(M)','SignFrequency(M)'],num_results=3):
    """
    Find all signs that match the given handshape.
    """
    # exclude the current sign from the search
    matches = signdata[(signdata['Handshape.2.0'] == handshape) & (signdata['EntryID'] != entryID)]
    print(f"Found {len(matches)} matches for handshape '{handshape}'")
    print("new")
    if matches.empty:
        print("No matches found.")
        return None
    weight = 1.0 / len(sort_flag)
    # matches['ranking_score'] = matches.apply(lambda row: sum(row[flag] * weight for flag in sort_flag), axis=1)
    matches['ranking_score'] = matches[sort_flag].sum(axis=1) * weight
    # Step 2: Compute score percentiles
    matches['percentile'] = matches['ranking_score'].rank(pct=True)

    # Step 3: Categorize by percentile
    def categorize(p):
        if p >= 0.67:
            return 'A'  # top 33%
        elif p <= 0.33:
            return 'C'  # bottom 33%
        else:
            return 'B'  # middle 33%
    matches['category'] = matches['percentile'].apply(categorize)
    category_probs = {'A': 0.5, 'B': 0.3, 'C': 0.2}
    sampled = []
    for cat, group in matches.groupby('category'):
        if cat in category_probs:
            n = max(1, int(num_results * category_probs[cat]))
            sampled.append(group.sample(n=min(n, len(group))))
    # Step 5: Final sort and return
    result = pd.concat(sampled).sort_values(by='ranking_score', ascending=False).head(num_results)


    # # sort the matches by the specified columns
    # matches = matches.sort_values('ranking_score', ascending=False)
    # # limit the number of results
    # matches = matches.head(num_results)
    print(result[['EntryID', 'LemmaID', 'Code', 'ranking_score','SignBankEnglishTranslations']])


    return result[['EntryID', 'LemmaID', 'Code', 'SignBankEnglishTranslations']]

def find_sign_by_movement(entryID,movement, repeated_movement, rotation,sort_flag='Iconicity(M)', num_results=3):
    """
    Find all signs that match the given location.
    """
    matches = signdata[(signdata['Movement.2.0'] == movement) &
                       (signdata['RepeatedMovement.2.0'] == repeated_movement) &
                       (signdata['UlnarRotation.2.0'] != rotation) &
                       (signdata['EntryID'] != entryID)]
    if matches.empty:
        return None
    # sort the matches by the specified columns
    matches = matches.sort_values(by=sort_flag, ascending=False)
    # limit the number of results
    matches = matches.head(num_results)

    return matches[['EntryID', 'LemmaID', 'Code', 'SignBankEnglishTranslations']]

def find_sign_by_location(entryID,majorloc, minorloc, sort_flag='Iconicity(M)', num_results=3):
    """
    Find all signs that match the given location.
    """
    matches = signdata[(signdata['MajorLocation.2.0'] == majorloc) &
                       (signdata['MinorLocation.2.0'] == minorloc) &
                       (signdata['EntryID'] != entryID)]
    if matches.empty:
        return None
    # sort the matches by the specified columns
    matches = matches.sort_values(by=sort_flag, ascending=False)
    # limit the number of results
    matches = matches.head(num_results)

    return matches[['EntryID', 'LemmaID', 'Code', 'SignBankEnglishTranslations']]

def find_sign_by_phono(entryID, similar_signs_code_set, sort_flag='num_matched_features', num_results=3):
    """
    Find all signs that match the given phonological properties.
    """
    # Determine which DataFrame to use based on the first letter of the code
    codeID = entry_code_check[entry_code_check['EntryID'] == entryID]['Code'].values
    if codeID.size > 0:
        codeID = codeID[0]
        code_first_letter = codeID[0]
    else:
        return None
    if code_first_letter in phono_name_list:
        phono_df = phono_list[phono_name_list.index(code_first_letter)]
        # find matches and sort by the specified columns, and limit the number of results
        matches = phono_df[phono_df['target'] == codeID]
        if matches.empty:
            return None
        # exclude the current code from the results
        matches = matches[~matches['neighbor'].isin(similar_signs_code_set)]
        matches = matches.sort_values(by=sort_flag, ascending=False)
        matches = matches.head(num_results)
        
        similar_signs_num_features = matches['num_matched_features'].values
        similar_signs_matched_features = matches['matched_features'].values
        similar_signs_code = matches['neighbor'].values.tolist()
        similar_signs_entry = entry_code_check[entry_code_check['Code'].isin(similar_signs_code)]['EntryID'].values
        similar_signs_lemma = entry_code_check[entry_code_check['Code'].isin(similar_signs_code)]['LemmaID'].values

        print(f"Similar signs for entryID '{entryID}' with code '{codeID}': {similar_signs_entry},\n{similar_signs_lemma},\n {similar_signs_code},\n {similar_signs_num_features},\n {similar_signs_matched_features}")
        return similar_signs_entry,similar_signs_lemma,similar_signs_code
    else:
        print(f"No phonological data available for code starting with '{code_first_letter}'")
        return None
    
def get_sign_info(entryID):
    sign_info = dict()
    if entryID.startswith('L_'):
        entryID = entryID[2:]
    sign_row = signdata[signdata['EntryID'] == entryID]
    print(f"utils-get_sign_info: EntryID: {entryID}, Sign Row: {sign_row}")
    if 'Handshape.2.0' not in sign_row.columns or sign_row.empty:
        print(f"No sign information found for EntryID: {entryID}")
        return None
    print("sign_row['Handshape.2.0'].values",sign_row['Handshape.2.0'].values)
    sign_info['handshape'] = sign_row['Handshape.2.0'].values[0]
    sign_info['movement'] = sign_row['Movement.2.0'].values[0]
    sign_info['repeated_movement'] = sign_row['RepeatedMovement.2.0'].values[0]
    sign_info['rotation'] = sign_row['UlnarRotation.2.0'].values[0]
    sign_info['major_location'] = sign_row['MajorLocation.2.0'].values[0]
    sign_info['minor_location'] = sign_row['MinorLocation.2.0'].values[0]

    return sign_info


#%%
def find_sign(entryID, aspect='handshape',num_results=3):
    """
    Find signs based on the specified aspect.
    """
    print(f"utils-find_sign: Finding signs for entryID: {entryID}, aspect: {aspect}, num_results: {num_results}")
    similar_signs_entryID = []
    similar_signs_lemmaID = []
    similar_signs_codes = []
    assert not entryID.startswith('L_')

    # find sign in signdata
    sign_row = signdata[signdata['EntryID'] == entryID]
    
    print(f"Start: Finding signs for entryID: {entryID}, aspect: {aspect}, num_results: {num_results}")
    if aspect == 'handshape':
        target_handshape = sign_row['Handshape.2.0'].values[0]
        print(f"Target handshape for entryID '{entryID}': {target_handshape}")
        matches = find_sign_by_handshape(entryID, target_handshape, num_results=num_results)
        print(f"Matches found for handshape '{target_handshape}':\n{matches}")
        if matches is not None:
            similar_signs_entryID.extend(matches['EntryID'].values)
            similar_signs_lemmaID.extend(matches['LemmaID'].values)
            similar_signs_codes.extend(matches['Code'].values)
    elif aspect == 'movement':
        target_movement = sign_row['Movement.2.0'].values[0]
        target_repeated_movement = sign_row['RepeatedMovement.2.0'].values[0]
        target_rotation = sign_row['UlnarRotation.2.0'].values[0]
        print(f"Target movement for entryID '{entryID}': {target_movement}, repeated movement: {target_repeated_movement}, rotation: {target_rotation}")
        matches = find_sign_by_movement(entryID, target_movement, target_repeated_movement, target_rotation, num_results=num_results)
        print(f"Matches found for movement '{target_movement}', repeated movement '{target_repeated_movement}', rotation '{target_rotation}':\n{matches}")
        if matches is not None:
            similar_signs_entryID.extend(matches['EntryID'].values)
            similar_signs_lemmaID.extend(matches['LemmaID'].values)
            similar_signs_codes.extend(matches['Code'].values)
    elif aspect == 'location':
        target_major_location = sign_row['MajorLocation.2.0'].values[0]
        target_minor_location = sign_row['MinorLocation.2.0'].values[0]
        matches = find_sign_by_location(entryID, target_major_location, target_minor_location, num_results=num_results)
        if matches is not None:
            similar_signs_entryID.extend(matches['EntryID'].values)
            similar_signs_lemmaID.extend(matches['LemmaID'].values)
            similar_signs_codes.extend(matches['Code'].values)
    elif aspect == 'mix':
        similar_signs_entry,similar_signs_lemma,similar_signs_code = find_sign_by_phono(entryID, similar_signs_codes, num_results=num_results)
        if similar_signs_entry is not None:
            similar_signs_entryID.extend(similar_signs_entry)
            similar_signs_lemmaID.extend(similar_signs_lemma)
            similar_signs_codes.extend(similar_signs_code)
    
    if len(similar_signs_entryID) < num_results:
        similar_signs_entry,similar_signs_lemma,similar_signs_code = find_sign_by_phono(entryID, similar_signs_codes, num_results=num_results - len(similar_signs_entryID))
        if similar_signs_entry is not None:
            similar_signs_entryID.extend(similar_signs_entry)
            similar_signs_lemmaID.extend(similar_signs_lemma)
            similar_signs_codes.extend(similar_signs_code)

    print(f"utils-find_sign: Similar signs for entryID '{entryID}' in aspect '{aspect}':")
    for i in range(len(similar_signs_entryID)):
        print(f"EntryID: {similar_signs_entryID[i]}, LemmaID: {similar_signs_lemmaID[i]}, Code: {similar_signs_codes[i]}")
    return similar_signs_entryID, similar_signs_lemmaID

#%%
def generate_story(input_words, num_sentences=1, num_words=7):
    prompt_user = f"Generate a short story using the following words: {input_words}. The story should be very simple, with short, clear sentences. The story should be engaging and easy to understand.\n"
    prompt_user += "Example Input: leaf, tree. 1 sentences, 7 words.\n"
    prompt_user += "Example Output: The leaves on the tree are colorful.\n"
    prompt_user += "Input: " + ', '.join(input_words) + f". Limit the story to {num_sentences} sentences, each under {num_words} words. Output:\n"
    prompt_system = ""
    story = gpt_run(prompt_system, prompt_user)
    
    return story

#%%
# # Example sentence
# sentence = "Where is the restaurant?"
# asl_list, entry_list = get_gloss_from_sentence(sentence)
# print("ASL Glosses:", asl_list)
# print("Entry IDs:", entry_list)

# # %%
# entry_sample = entry_list[0]  # Example entry ID
# result = find_sign('restaurant', aspect='handshape')
# print(result)
# print()
# result = find_sign('restaurant', aspect='movement')
# print(result)
# print()
# result = find_sign('restaurant', aspect='location')
# print(result)
# print()
# result = find_sign('restaurant', aspect='phonology')
# print(result)
# print()
 