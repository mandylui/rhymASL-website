#%%
# !/opt/homebrew/bin/python3.9
# -*- coding: utf-8 -*-
"""
This module provides the text to gloss translation system using gpt-3.5 turbo.
"""
from openai import OpenAI
from dotenv import load_dotenv
import os
import pandas as pd
import contractions
from num2words import num2words
import re


load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def gpt_run(prompt_system: str, prompt_user: str):
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": prompt_system},
            {"role": "user", "content": prompt_user}
        ]
    )
    print(completion.choices[0].message)
    return completion.choices[0].message.content


"""
This function generates the prompt for the user to obtain ASL gloss from english text
input: new_sentence (string)
return: the tuple (prompt for the system (string), prompt for the user (string))
"""
def generate_prompt_wo_nms(new_sentence):
    system_prompt = ""
    user_prompt = ""
    user_prompt += """You are a native signer with expertise in English to American Sign Language (ASL) translation. Translate English sentences into ASL gloss. Output an empty string if none is returned. Expected output format is ASL: "[ASL translation here]". No additional explanations are needed, only provide the ASL gloss following specific grammar rules:"""
    user_prompt += """1. **Word Order**: ASL follows a Time-Topic-Comment structure. Establish the time frame first, then the subject, and finally the comment about the subject.
   - Example: English: 'I went to the store yesterday'; ASL: "YESTERDAY STORE I GO"

2. **No Articles or 'Be' Verbs**: ASL does not use articles like 'a,' 'an,' 'the,' or 'be' verbs like 'am,' 'is,' 'are,' 'was,' or 'were.'
   - Example: English: 'I saw a dog'; ASL: "DOG I SEE"
   - Example: English: 'She is happy'; ASL: "SHE HAPPY"

3. **Spatial relationships**: Describe the relationship between objects by placing them in the order of object first, then the subject and preposition. Do not remove prepositions such as 'on,' 'in,' 'from,' or 'under.'
  - Example: English: "The book is on the table."; ASL: "TABLE BOOK ON"

4. **WH-Questions**: WH-words (WHO, WHAT, WHERE, WHEN, WHY, HOW, and WHICH) are placed at the end of the sentence. Use appropriate facial expressions such as furrowing your eyebrows.
   - Example:  English: 'What is your name?'; ASL: "YOUR NAME WHAT"

5. **Yes/No Questions**: For yes/no questions, raise your eyebrows and lean slightly forward.
   - Example: English: 'Do you like pizza?'; ASL: "PIZZA YOU LIKE"

6. **Negation**: To negate a sentence, use the negation sign after the word it negates, and shake your head.
   - Example: English: 'I don't have a car.'; ASL: "ME NONE CAR"
   - Example: English:  'I am not a student.'; ASL: “ME NOT STUDENT ME”
   - Example: English: 'I don't understand your idea.'; ASL: “YOUR IDEA ME NOT UNDERSTAND”

7. **Possessive and Demonstrative Determiners with Objects Connection**: Keep possessive determiners (e.g. 'your' 'my,' etc) or demonstrative determiners (e.g., 'this,' 'that,' 'these,' 'those,' and 'many) connected to the nouns they describe or introduce. Ensure possessive determiners (e.g. 'your' 'my,' etc) or demonstrative determiners (e.g., 'this,' 'that,' 'these,' 'those,' and 'many) are not removed
  - Example: English: 'Do you think my shoes are ugly?'; ASL: "MY SHOES YOU THINK UGLY"
  - Example: English: 'Do you like this car?'; ASL: “THIS CAR YOU LIKE”
    
8. **Handle Verb Sequences**: Maintain orders between main verb and it's clausal complement in sentences, such as 'want to go,' 'need to go,' 'get going,' 'try moving,' 
   - Example: English:  'Where do you want to go?'; ASL: “YOU WANT GO WHERE”
   - Example: English: 'Try moving the car'; ASL: "TRY MOVE CAR"

9. **Maintain noun phrase and verb phrase Order**: Keep the word order intact for phrases involving pronouns with verbs, verbs with pronouns, or verbs followed by adverbs.
  - Example: English: 'I ran fast'; ASL: "I RUN FAST"
  - Example: English: 'give me the key'; ASL: “KEY GIVE ME”
   
10. **Adjective-Noun Order**: When an adjective comes before the noun it describes, retain the order of adjective and noun.
  - Example: English: 'She drives a blue car'; ASL: "CAR BLUE SHE DRIVE"
   
11. **Keep verb Phrase Order Intact**: Ensure that verb phrases like 'move around,' 'take away' are kept together and not split.
  - Example: English: 'We move around the floor gracefully'; ASL: "WE MOVE AROUND FLOOR GRACEFUL"

Translate the following sentences into ASL and NMS following these rules:"""

    user_prompt += f"New sentence: \"{new_sentence}\""

    # print(system_prompt, user_prompt)
    return (system_prompt, user_prompt)

#%%
def parsing_gpt_response_wo_nms(response):
    print(response)
    if len(response) == 0:
        return ""
    # find two quotes and split the string
    split_response = response.split('"')
    if len(split_response) >1:
        asl = split_response[1]
    else:
        asl = response
    return asl

def generate_asl_gloss_wo_nms(new_sentence):
    system_prompt, user_prompt = generate_prompt_wo_nms(new_sentence)
    asl_nms = gpt_run(system_prompt, user_prompt)
    asl = parsing_gpt_response_wo_nms(asl_nms)
    return asl


# Function to clean text before translation
def clean_text(text):
    if not isinstance(text, str):  # Check if text is not a string
        return ""
    
    # Step 0: lowercase the text
    text = text.lower()

    # Step 1: Expand contractions
    expanded_text = contractions.fix(text)

    # Step 2: Remove "there be" if it appears at the beginning of the sentence
    pattern = r'^\s*there\s+(is|are|was|were)\s+'
    expanded_text = re.sub(pattern, '', expanded_text, flags=re.IGNORECASE).strip()


    # Step 3: Remove "let" if it appears at the beginning of the sentence
    expanded_text = re.sub(r'^\blet\b', '', expanded_text, flags=re.IGNORECASE).strip()

    # Step 4: Convert numbers to words and replace hyphens with spaces
    expanded_text = re.sub(r'\b\d+\b', lambda x: num2words(int(x.group(0))).replace('-', ' '), expanded_text)

    # expanded_text = expanded_text.replace(" it ", " this ")
    # Step 5: replace "it" if it appears at the beginning of the sentence
    expanded_text = re.sub(r'\b(it)\b', 'this', expanded_text, flags=re.IGNORECASE)
    expanded_text = re.sub(r'\b(in the front)\b', 'front', expanded_text, flags=re.IGNORECASE)
    expanded_text = re.sub(r'\b(in front)\b', 'front', expanded_text, flags=re.IGNORECASE)

    # expanded_text = expanded_text.replace(" the ", " this ")

    expanded_text = expanded_text.replace("a number of", "many")
    expanded_text = expanded_text.replace("a lot of", "many")
    expanded_text = expanded_text.replace("a lot", "much")

    # expanded_text = expanded_text.replace(" i ", " me ")
    expanded_text = re.sub(r'\b(i)\b', 'me', expanded_text, flags=re.IGNORECASE)

    if any(word in expanded_text.lower() for word in ["which", "what", "how", "why", "when", "where", "who"]):
        expanded_text = expanded_text.strip() + "?"


    return expanded_text


# Function to clean text before translation
def clean_gloss(gloss):
    if not isinstance(gloss, str):  # Check if text is not a string
        return ""
    
    # Step 0: lowercase the text
    cleaned_gloss = gloss.lower()

    # Step 1: replace some fixed phrases
    cleaned_gloss = re.sub(r'\b(look like)\b', 'look-like', cleaned_gloss, flags=re.IGNORECASE)
    cleaned_gloss = re.sub(r'\b(not understand)\b', 'not-understand', cleaned_gloss, flags=re.IGNORECASE)
    cleaned_gloss = re.sub(r'\b(set up)\b', 'set-up', cleaned_gloss, flags=re.IGNORECASE)
    cleaned_gloss = re.sub(r'\b(go ahead)\b', 'go-ahead', cleaned_gloss, flags=re.IGNORECASE)
    cleaned_gloss = re.sub(r'\b(look for)\b', 'look-for', cleaned_gloss, flags=re.IGNORECASE)

    return cleaned_gloss

# Function to process each sentence and generate its gloss
def translate_sentence_to_gloss(sentence):
    # Clean the sentence
    cleaned_sentence = clean_text(sentence)

    # Translate the cleaned sentence to ASL gloss
    gloss = generate_asl_gloss_wo_nms(cleaned_sentence)

    # Clean the gloss
    cleaned_gloss = clean_gloss(gloss)

    return cleaned_gloss

# Main function to process the CSV file
def process_csv(input_csv_path, output_csv_path):
    # Load the CSV with original sentences
    df = pd.read_csv(input_csv_path)

    # Print the columns to verify column names
    print("Columns in the CSV file:", df.columns)

    # Check if 'sentence' column exists
    if 'sentences' not in df.columns:
        print("Error: 'sentence' column not found. Please check the CSV file.")
        return
    # Drop duplicate sentences
    df = df.drop_duplicates(subset=['sentences'])

    # Print the number of unique sentences
    print(f"Number of unique sentences: {len(df)}")

    # Filter rows to process a subset, if necessary
    df = df.iloc[0:400]
    # Apply the translate function to each sentence
    df['gloss'] = df['sentences'].apply(translate_sentence_to_gloss)

    # Save to a new CSV file
    df.to_csv(output_csv_path, index=False)

    print(f"Gloss translation saved to {output_csv_path}")

def translate_to_gloss(text: str) -> str:
    cleaned = clean_text(text)
    gloss = generate_asl_gloss_wo_nms(cleaned)
    return clean_gloss(gloss)

# Example usage
# input_csv_path = "text_to_gloss_testing/50_sentences2.csv"
# output_csv_path = "text_to_gloss_testing/50_sentences2_output_phrase.csv"

# # Run the process
# process_csv(input_csv_path, output_csv_path)

#%%
# sample usage
# print(correct_ocr_text("Want to use the knife?"))


# new_sentence = "I am going to the store"
# asl, nms = generate_asl_gloss(new_sentence)
# asl_list = asl.split(" ")
# print(asl_list)

# new_sentence = "Where is the restaurant?"
# asl, nms = generate_asl_gloss(new_sentence)
# asl_list = asl.split(" ")
# print(asl_list)

# new_sentence = "I am going to the store"
# asl = generate_asl_gloss_wo_nms(new_sentence)
# asl_list = asl.split(" ")
# print(asl_list)

# new_sentence = "Where is the restaurant?"
# asl = generate_asl_gloss_wo_nms(new_sentence)
# asl_list = asl.split(" ")
# print(asl_list)