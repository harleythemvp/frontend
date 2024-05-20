import torch
import nltk
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForMaskedLM
import joblib
from fuzzywuzzy import fuzz


tokenizer_mlm = AutoTokenizer.from_pretrained("zklmorales/bert_mlm_fine-tuned")
model_mlm = AutoModelForMaskedLM.from_pretrained("zklmorales/bert_mlm_fine-tuned")

# Load pre-trained BERT tokenizer and model for sequence classification
tokenizer_cls = AutoTokenizer.from_pretrained("zklmorales/bert_finetuned")
model_cls = AutoModelForSequenceClassification.from_pretrained("zklmorales/bert_finetuned")

# Load CRF Model for POS Tagging
crf_model = joblib.load(r'C:\Users\edwar\OneDrive\Desktop\thesis_v2\server\crf_model.pkl')

# Define function to extract word features
def word_features(sent, i):
    word = sent[i][0]
    pos = sent[i][1]
    
    # first word
    if i == 0:
        prevword = '<START>'
        prevpos = '<START>'
    else:
        prevword = sent[i-1][0]
        prevpos = sent[i-1][1]
        
    # first or second word
    if i == 0 or i == 1:
        prev2word = '<START>'
        prev2pos = '<START>'
    else:
        prev2word = sent[i-2][0]
        prev2pos = sent[i-2][1]
    
    # last word
    if i == len(sent) - 1:
        nextword = '<END>'
        nextpos = '<END>'
    else:
        nextword = sent[i+1][0]
        nextpos = sent[i+1][1]
    
    # suffixes and prefixes
    pref_1, pref_2, pref_3, pref_4 = word[:1], word[:2], word[:3], word[:4]
    suff_1, suff_2, suff_3, suff_4 = word[-1:], word[-2:], word[-3:], word[-4:]
    
    return {'word':word,            
            'prevword': prevword,
            'prevpos': prevpos,  
            'nextword': nextword, 
            'nextpos': nextpos,          
            'suff_1': suff_1,  
            'suff_2': suff_2,  
            'suff_3': suff_3,  
            'suff_4': suff_4, 
            'pref_1': pref_1,  
            'pref_2': pref_2,  
            'pref_3': pref_3, 
            'pref_4': pref_4,
            'prev2word': prev2word,
            'prev2pos': prev2pos           
           }


def grammar_check(new_sentence):
    # Tokenize the new sentence and get POS tags
    tokens = nltk.word_tokenize(new_sentence)
    tagged_tokens = [nltk.pos_tag([token])[0] for token in tokens]

    # Extract features for each token in the new sentence
    features = [word_features(tagged_tokens, i) for i in range(len(tagged_tokens))]

    # Use the BERT classifier to check if the sentence is grammatically correct
    inputs_cls = tokenizer_cls(new_sentence, return_tensors="pt")
    with torch.no_grad():
        outputs_cls = model_cls(**inputs_cls)
    probabilities_cls = torch.softmax(outputs_cls.logits, dim=1).squeeze().tolist()
    predicted_class = torch.argmax(outputs_cls.logits, dim=1).item()

    grammar_correction_candidates = []

    # Check if the sentence is grammatically correct
    if predicted_class == 1:
        return 'Ang pangungusap ay wasto.'
    else:
        # Use the CRF model to predict POS tags for the tokens
        predicted_labels = crf_model.predict([features])[0]

        # Combine tokens with predicted labels
        predicted_tokens_with_labels = list(zip(tokens, predicted_labels))

        print("Original sentence:", new_sentence)

        

        # Iterate over each word and mask it, then predict the masked word
        for i, (token, predicted_label) in enumerate(zip(tokens, predicted_labels)):
            # Check if the predicted label is a verb
            if predicted_label.startswith('VB'):
                # Mask the word
                masked_words = tokens.copy()
                masked_words[i] = tokenizer_mlm.mask_token
                masked_sentence = " ".join(masked_words)

                # Tokenize the masked sentence
                tokens_mlm = tokenizer_mlm(masked_sentence, return_tensors="pt")

                # Get the position of the masked token
                masked_index = torch.where(tokens_mlm["input_ids"] == tokenizer_mlm.mask_token_id)[1][0]

                # Get the logits for the masked token
                with torch.no_grad():
                    outputs = model_mlm(**tokens_mlm)
                    predictions_mlm = outputs.logits

                # Get the top predicted words for the masked token
                top_predictions_mlm = torch.topk(predictions_mlm[0, masked_index], k=10)
                candidates_mlm = [tokenizer_mlm.decode(idx.item()) for idx in top_predictions_mlm.indices]

                # Reconstruct the sentence with each candidate
                for candidate_mlm in candidates_mlm:
                    # Get embeddings for the masked word and the candidate word
                    original_embedding = model_mlm.get_input_embeddings()(torch.tensor(tokenizer_mlm.encode(token, add_special_tokens=False))).mean(dim=0)
                    candidate_embedding = model_mlm.get_input_embeddings()(torch.tensor(tokenizer_mlm.encode(candidate_mlm, add_special_tokens=False))).mean(dim=0)
                    
                    # Compute cosine similarity between original masked word and predicted word
                    similarity = torch.nn.functional.cosine_similarity(original_embedding.unsqueeze(0), candidate_embedding.unsqueeze(0)).item()
                    fuzzy_match_score = fuzz.ratio(token, candidate_mlm)

                    replaced_words = masked_words.copy()
                    replaced_words[i] = candidate_mlm
                    corrected_sentence = " ".join(replaced_words).split()  # Split and join to remove extra spaces
                    corrected_sentence = " ".join(corrected_sentence)  # Join words without extra spaces
                    
                    # Tokenize the corrected sentence for sequence classification
                    inputs_cls = tokenizer_cls(corrected_sentence, return_tensors="pt")

                    # Forward pass through the model for sequence classification 1 or 0 
                    with torch.no_grad():
                        outputs_cls = model_cls(**inputs_cls)

                    # Get softmax probabilities for class indicating grammatically correct sentences
                    probability = torch.softmax(outputs_cls.logits, dim=1).squeeze().tolist()[1]
                    probability = round(probability, 3)

                    # Append the corrected sentence along with its probability and cosine similarity
                    grammar_correction_candidates.append((corrected_sentence, probability, similarity, fuzzy_match_score))


        # Sort the grammar correction candidates by their probabilities and cosine similarities in descending order
        grammar_correction_candidates.sort(key=lambda x: (x[1], x[3], x[2]), reverse=True)

    if grammar_correction_candidates:
        candidate, probability, cosine_similarity, fuzzy_match_score = grammar_correction_candidates[0]
        print("Sentence:", candidate)
        print("Correctness Probability:", probability)
        print("Cosine Similarity:", cosine_similarity)
        print("Levenshtein Score:", fuzzy_match_score)
    
    if(len(grammar_correction_candidates) == 0):
        return 'Walang solusyong nailabas.'
    return candidate