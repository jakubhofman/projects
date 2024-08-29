from gensim.models import Word2Vec, Phrases
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models import KeyedVectors
from gensim.models.phrases import Phraser
from gensim.utils import simple_preprocess

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import log_loss


import re

import numpy as np
import pandas as pd
import scipy


import xgboost as xgb
import catboost as ctb


# DATA
train = pd.read_csv('/kaggle/input/lmsys-chatbot-arena/train.csv')
test = pd.read_csv('/kaggle/input/lmsys-chatbot-arena/test.csv')
most_common_english_words = open("/kaggle/input/most-common-words/most_common_english_words.txt").read().split()


# PREPARE DATA
def run_eval_for_string(df, cols):
    for col in cols:
        df[col] = df[col].map(lambda x: x if isinstance(x, list) else eval(x.replace("null", "None")))
    
def replace_na(docs):
    return [doc if doc else "NONE" for doc in docs]

def flat_text_arr(docs):
    if isinstance(docs, list):
        return " ".join([doc for doc in replace_na(docs)])
    return docs

TARGETS = ["winner_model_a", "winner_model_b", "winner_tie"]
cols = ["prompt", "response_a", "response_b"]

for col in cols:
    train[f"{col}_raw"] = train[col]

    run_eval_for_string(train, cols)

conditions = [ (train[x] == 1) for x in TARGETS]
train['winner'] = np.select(conditions, TARGETS, default='none')
train['target'] = np.where(train[TARGETS])[1]

full_data = train.copy()
for col in ['prompt', 'response_a', 'response_b']:
    full_data[col] = full_data[col].map(flat_text_arr)
    
full_corpus = pd.concat([full_data['prompt'], 
                         full_data['response_a'], 
                         full_data['response_b']], 
                         ignore_index=True)


# TOKENS
full_corpus_simple_token = full_corpus.map(simple_preprocess)


# WORD2VEC
w2v_model = Word2Vec(full_corpus_simple_token, window=5, vector_size=300, sg=1, hs=0, seed=0, min_count=2, workers=4) 

def document_vector(doc):
    doc = [word for word in simple_preprocess(doc) if word in w2v_model.wv]
    if len(doc) == 0:
        return np.zeros(w2v_model.vector_size)
    return np.mean(w2v_model.wv[doc], axis=0)


# TRANSFORM TRAIN DATASET
cols_to_vec = ["prompt","response_a", "response_b"]
vec_columns = []

for col in cols_to_vec:
    vec_columns.append(np.array([document_vector(text) for text in train[col].map(flat_text_arr)]))
    
combined_w2v = np.hstack(vec_columns)



# FUNCTIONS - feature engineering
def _contains_pattern_in(text, pattern):
    if text is None: return 0
    return 1 if re.findall(pattern, text, re.IGNORECASE) else 0


def _contains_code_patterns(text):
    if pd.isna(text):
        return 0
    # Check for presence of code patterns
    patterns = ["def ", "print(", "import ", "#include ", "println", "fn ", "namespace ", "CREATE TABLE ",
                "ALTER TABLE"]
    return 1 if any(pattern in text for pattern in patterns) else 0

def _count_unique_words(text):
    if text is None:
        return 0
    return len(set(text.split()))


def _len(val):
    return 0 if val is None else len(val)


def _log1p(val):
    return np.log(val + 1)


def _words(val):
    return "" if val is None else val.split()


def _most_common_words(val_1, val_2):
    numb = 0
    for i in range(0, val_1):
        if val_2[i].lower() in most_common_english_words:
            numb += 1
    return numb


def _bold(text):
    if pd.isna(text):
        return 0
    patterns = ["**"]
    return 1 if any(pattern in text for pattern in patterns) else 0

def _is_text_neutral(text):
    if text is None:
        return None
    formal_words = ['Dear', 'Sir', 'Madam', 'Sincerely', 'Regards']
    informal_words = ['Hey', 'Hi', 'Hello', 'Thanks', 'Cheers']
    formal_score = sum([text.count(word) for word in formal_words])
    informal_score = sum([text.count(word) for word in informal_words])
    if formal_score > informal_score:
        return 0
    elif informal_score > formal_score:
        return 0
    else:
        return 1


def _looking_for_points(text):
    if text is None: return 0

    lists = re.findall(r'\n\d+\.\s.*?(?=\n\n|\n\d+\.|\Z)', text, re.DOTALL)
    return len(lists)


def _count_question_marks(text):
    if text is None:
        return 0  # Return 0 if text is None
    return text.count('?')


def _count_exclamation_marks(text):
    if text is None:
        return 0  # Return 0 if text is None
    return text.count('!')


def _common_words(text1, text2):
    if (text1 is None) | (text2 is None):
        return {}
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    common_words = words1.intersection(words2)
    return common_words


def _ratio_comm_words_res_a_res_b(response_a, response_b):
    if (response_a is None) | (response_b is None):
        return 0
    else:
        return _count_common_words(response_a, response_b) / (
                    _count_unique_words(response_a) + _count_unique_words(response_b))


def _combine_unique_words(series):
    result_set = set()
    for word_list in series:
        result_set.update(word_list)
    return len(result_set)


def _analyze_triple(id_, id_row, prompt, response_a, response_b):
    len_prompt = _len(prompt)
    len_log_prompt = _log1p(len_prompt)
    len_res_a = _len(response_a)
    len_log_res_a = _log1p(len_res_a)
    len_res_b = _len(response_b)
    len_log_res_b = _log1p(len_res_b)  
    apologetic_keywords = r'I apologize|sorry|unfortunately'
    apologetic_words_a = _contains_pattern_in(response_a, apologetic_keywords)
    apologetic_words_b = _contains_pattern_in(response_b, apologetic_keywords)
    len_unique_words_res_a = _count_unique_words(response_a)
    len_unique_words_res_b = _count_unique_words(response_b)
    len_unique_words_prompt = _count_unique_words(prompt)
    math_keywords = r'\b\d+[\+\-\*/^]\d+|\b\d+[\+\-\*/^]\d+[\+\-\*/^]\d+|\b\d+(\.\d+)?[+\-*/^]\d+(\.\d+)?|[\d\(\)\+\-\*/^\.]+=[\d\(\)\+\-\*/^\.]+'
    math_prompt = _contains_pattern_in(prompt, math_keywords)
    code_in_res_a = _contains_code_patterns(response_a)
    code_in_res_b = _contains_code_patterns(response_b)
    to_assist_in_res_b = _contains_pattern_in(response_b, "to assist you")
    ethical_in_res_a = _contains_pattern_in(response_a, "ethical")
    ethical_in_res_b = _contains_pattern_in(response_b, "ethical")
    summarize_pattern = r'summarize|summary|in summary|o sum up|in conclusion|to conclude|overall|in essence'
    summarize_in_prompt = _contains_pattern_in(prompt, summarize_pattern)
    reinforce_pattern = r'ultimately|thus|therefore|as a result|consequently'
    reinforce_in_res_a = _contains_pattern_in(response_a, reinforce_pattern)
    reinforce_in_res_b = _contains_pattern_in(response_b, reinforce_pattern)
    count_words_prompt = _len(_words(prompt))
    count_words_res_a = _len(_words(response_a))
    count_words_res_b = _len(_words(response_b))
    count_common_words_prompt = _most_common_words(_len(_words(prompt)), _words(prompt))
    count_common_words_res_a = _most_common_words(_len(_words(response_a)), _words(response_a))
    count_common_words_res_b = _most_common_words(_len(_words(response_b)), _words(response_b))
    bold_in_res_a = _bold(response_a)
    bold_in_res_b = _bold(response_b)

    return {

        "id": id_,
        "id_row": id_row,

        # top 50
        "diff_unique_words_res_b_prompt": len_unique_words_res_b - len_unique_words_prompt,
        "apologetic_word_ratio_res_a_b": apologetic_words_a / apologetic_words_b if apologetic_words_b > 0 else 0,
        "diff_unique_words_res_a_prompt": len_unique_words_res_a - len_unique_words_prompt,
        "diff_aplogetic_words_res_a_b": apologetic_words_a - apologetic_words_b,
        "is_apologetic_words_res_b": apologetic_words_b,
        "is_apologetic_words_res_a": apologetic_words_a,
        "diff_log_res_b_prompt": len_log_res_b - len_log_prompt,
        "diff_log_res_a_prompt": len_log_res_a - len_log_prompt,
        "len_unique_words_res_b": len_unique_words_res_b,
        "len_unique_words_res_a": len_unique_words_res_a,
        "len_log_res_b": len_log_res_b,
        "list_points_prompt": _looking_for_points(prompt),
        'reinforc_msg_res_a': reinforce_in_res_a,
        "ratio_res_b_prompt": len_res_b / len_prompt,
        "ratio_res_a_prompt": len_res_a / len_prompt,
        'reinforc_msg_res_b': reinforce_in_res_b,
        "len_log_res_a": len_log_res_a,
        "diff_res_a_prompt": len_res_a - len_prompt,
        "diff_res_b_prompt": len_res_b - len_prompt,
        'summarize_in_prompt': summarize_in_prompt,
        'unique_words_ratio_res_a': len_unique_words_res_a / len_unique_words_prompt if len_unique_words_prompt > 0 else 0,
        "count_exclamation_mark_res_b": _count_exclamation_marks(response_b),
        'unique_words_ratio_res_b': len_unique_words_res_b / len_unique_words_prompt if len_unique_words_prompt > 0 else 0,
        "to_assist_in_res_b": to_assist_in_res_b,
        "code_in_res_b": code_in_res_b,
        "code_in_res_a": code_in_res_a,
        "math_prompt": math_prompt,
        'ethical_in_res_a': ethical_in_res_a,
        'ethical_in_res_b': ethical_in_res_b,
        "count_question_mark_res_a": _count_question_marks(response_a),
        "res_a_neutral": _is_text_neutral(response_a),
        "res_b_neutral": _is_text_neutral(response_b),
        "len_res_a": len_res_a,
        "len_res_b": len_res_b,
        "common_words_res_a_res_b": _common_words(response_a, response_b),
        "list_points_res_a": _looking_for_points(response_a),
        "list_points_res_b": _looking_for_points(response_b),     
        "count_words_prompt": count_words_prompt,
        "count_words_res_a": count_words_res_a,
        "count_words_res_b": count_words_res_b,    
        "ratio_common_words_prompt": count_common_words_prompt / count_words_prompt if count_words_prompt > 0 else 0,
        "ratio_common_words_res_a": count_common_words_res_a / count_words_res_a if count_words_res_a > 0 else 0,
        "ratio_common_words_res_b": count_common_words_res_b / count_words_res_b if count_words_res_b > 0 else 0,       
        "bold_in_res_a": bold_in_res_a,
        "bold_in_res_b": bold_in_res_b,
    }


def _analyze_battle(df_items):
    df_battle = df_items.groupby("id").agg(
        n_items=("id_row", len),

        #         winner=("winner", lambda x: x[0]),
        #         target=("target", lambda x: x[0]),
        # top 50
        all_abs_mean_diff_unique_words_res_b_prompt=("diff_unique_words_res_b_prompt", lambda x: np.abs(x).mean()),
        all_mean_apologetic_word_ratio_res_a_b=("apologetic_word_ratio_res_a_b", np.mean),
        all_abs_mean_diff_unique_words_res_a_prompt=("diff_unique_words_res_a_prompt", lambda x: np.abs(x).mean()),
        all_sum_diff_diff_aplogetic_words_res_a_b=("diff_aplogetic_words_res_a_b", sum),
        all_mean_apologetic_words_res_b=("is_apologetic_words_res_b", np.mean),
        all_mean_apologetic_words_res_a=("is_apologetic_words_res_a", np.mean),
        all_sum_diff_log_res_b_prompt=("diff_log_res_b_prompt", sum),
        all_sum_diff_log_res_a_prompt=("diff_log_res_a_prompt", sum),
        all_abs_mean_diff_aplogetic_words_res_a_b=("diff_aplogetic_words_res_a_b", lambda x: np.abs(x).mean()),
        all_sum_apologetic_words_res_b=("is_apologetic_words_res_b", sum),
        all_mean__unique_words_res_b=("len_unique_words_res_b", np.mean),
        all_mean__unique_words_res_a=("len_unique_words_res_a", np.mean),
        all_mean_len_log_res_b=("len_log_res_b", np.mean),
        all_abs_sum_diff_unique_words_res_a_prompt=("diff_unique_words_res_a_prompt", lambda x: np.abs(x).sum()),
        all_sum_list_points_prompt=("list_points_prompt", sum),
        all_mean_reinforc_msg_res_a=("reinforc_msg_res_a", np.mean),
        all_mean_ratio_res_b_prompt=("ratio_res_b_prompt", np.mean),
        all_mean_ratio_res_a_prompt=("ratio_res_a_prompt", np.mean),
        all_mean_reinforc_msg_res_b=("reinforc_msg_res_b", np.mean),
        all_sum_len_log_res_a=("len_log_res_a", sum),
        all_sum_len_log_res_b=("len_log_res_b", sum),
        all_std_apologetic_words_res_b=("is_apologetic_words_res_b", np.std),
        all_abs_mean_diff_res_a_prompt=("diff_res_a_prompt", lambda x: np.abs(x).mean()),
        all_abs_mean_diff_res_b_prompt=("diff_res_b_prompt", lambda x: np.abs(x).mean()),
        all_mean_summarize_in_prompt=("summarize_in_prompt", np.mean),
        all_mean_unique_words_ratio_res_a_prompt=("unique_words_ratio_res_a", np.mean),
        all_mean_len_log_res_a=("len_log_res_a", np.mean),
        all_abs_sum_diff_unique_words_res_b_prompt=("diff_unique_words_res_b_prompt", lambda x: np.abs(x).sum()),
        all_mean_list_points_prompt=("list_points_prompt", np.mean),
        all_mean_exclamation_mark_res_b=("count_exclamation_mark_res_b", np.mean),
        all_sum__unique_words_res_a=("len_unique_words_res_a", sum),
        all_sum__unique_words_res_b=("len_unique_words_res_b", sum),
        all_mean_neutral_res_b=("res_b_neutral", np.mean),
        all_mean_unique_words_ratio_res_b_prompt=("unique_words_ratio_res_b", np.mean),
        all_mean_neutral_res_a=("res_a_neutral", np.mean),
        all_mean_to_assist_in_res_b=("to_assist_in_res_b", np.mean),
        all_mean_code_in_res_a=("code_in_res_a", np.mean),
        all_mean_code_in_res_b=("code_in_res_b", np.mean),
        all_mean_math_in_prompt=("math_prompt", np.mean),
        all_mean_ethical_in_res_a=("ethical_in_res_a", np.mean),
        all_mean_ethical_in_res_b=("ethical_in_res_b", np.mean),
        all_std_len_log_res_b=("len_log_res_b", np.std),
        all_std_len_log_res_a=("len_log_res_a", np.std),
        all_std_apologetic_words_res_a=("is_apologetic_words_res_a", np.std),
        all_sum_question_mark_res_a=("count_question_mark_res_a", sum),
        all_sum_ratio_res_a_prompt=("ratio_res_a_prompt", sum),
        all_sum_ratio_res_b_prompt=("ratio_res_b_prompt", sum),
        all_sum_len_res_a=("len_res_a", sum),
        all_sum_len_res_b=("len_res_b", sum),
        all_uniqe_comm_words_res_a_res_b=("common_words_res_a_res_b", _combine_unique_words),
        all_mean_list_points_res_a=("list_points_res_a", np.mean),
        all_std_list_points_res_a=("list_points_res_a", np.std),
        all_mean_list_points_res_b=("list_points_res_b", np.mean),
        all_std_list_points_res_b=("list_points_res_b", np.std),      
        all_mean_count_words_prompt = ("count_words_prompt",np.mean),
        all_mean_count_words_res_a = ("count_words_res_a",np.mean),
        all_mean_count_words_res_b = ("count_words_res_b",np.mean),
        all_mean_ratio_common_words_res_a = ("ratio_common_words_res_a", np.mean),
        all_mean_ratio_common_words_res_b = ("ratio_common_words_res_b", np.mean),
        all_mean_bold_in_res_a = ("bold_in_res_a",np.mean),
        all_mean_bold_in_res_b = ("bold_in_res_b",np.mean),

    ).fillna(-1)

    df_battle = df_battle.reset_index()
    df_battle['all_res_a_shorter'] = (df_battle["all_sum_len_res_a"] < df_battle["all_sum_len_res_b"])
    df_battle['all_res_a_longer'] = (df_battle["all_sum_len_res_a"] > df_battle["all_sum_len_res_b"])
    df_battle['all_res_equal'] = (df_battle["all_sum_len_res_a"] == df_battle["all_sum_len_res_b"])
    df_battle["ratio_comm_words_res_a_res_b_in_res_a"] = df_battle["all_uniqe_comm_words_res_a_res_b"] / \
                                                         df_battle["all_sum_len_res_a"]
    df_battle["ratio_comm_words_res_a_res_b_in_res_b"] = df_battle["all_uniqe_comm_words_res_a_res_b"] / \
                                                         df_battle["all_sum_len_res_b"]

    return df_battle


def get_top_50_feats(df):
    items = []

    cols = ["id", "prompt", "response_a", "response_b"]

    vls = [df[x].values for x in cols]

    for id_, prompt_ls, response_a_ls, response_b_ls in zip(*vls):
        for id_row, (prompt, response_a, response_b) in enumerate(zip(prompt_ls, response_a_ls, response_b_ls)):
            items.append(_analyze_triple(id_, id_row, prompt, response_a, response_b))

    df_items = pd.DataFrame(items)
    df_battle = _analyze_battle(df_items)
    return df_battle[[
        "id", "all_res_a_shorter", "all_res_a_longer", "all_abs_mean_diff_unique_words_res_b_prompt",
        "all_mean_apologetic_word_ratio_res_a_b", "all_abs_mean_diff_unique_words_res_a_prompt",
        "all_sum_diff_diff_aplogetic_words_res_a_b", "all_res_equal", "all_mean_apologetic_words_res_b",
        "ratio_comm_words_res_a_res_b_in_res_a", "all_mean_apologetic_words_res_a", "all_sum_diff_log_res_b_prompt",
        "all_sum_diff_log_res_a_prompt", "all_sum_len_log_res_b", "all_mean_ratio_res_a_prompt",
        "all_abs_mean_diff_aplogetic_words_res_a_b", "ratio_comm_words_res_a_res_b_in_res_b",
        "all_sum_apologetic_words_res_b", "all_mean__unique_words_res_b", "all_mean__unique_words_res_a",
        "all_mean_len_log_res_b", "all_abs_sum_diff_unique_words_res_a_prompt", "all_sum_list_points_prompt",
        "all_mean_reinforc_msg_res_a", "all_mean_ratio_res_b_prompt", "all_mean_reinforc_msg_res_b",
        "all_sum_len_log_res_a", "all_std_apologetic_words_res_b", "all_abs_mean_diff_res_a_prompt", 
        "all_abs_mean_diff_res_b_prompt","all_mean_summarize_in_prompt", "all_mean_unique_words_ratio_res_a_prompt",
        "all_mean_len_log_res_a","all_abs_sum_diff_unique_words_res_b_prompt", "all_mean_list_points_prompt",
        "all_mean_exclamation_mark_res_b","all_sum__unique_words_res_a", "all_sum__unique_words_res_b", "all_mean_neutral_res_b",
        "all_mean_unique_words_ratio_res_b_prompt", "all_mean_neutral_res_a", "all_mean_to_assist_in_res_b",
        "all_mean_code_in_res_a", "all_mean_code_in_res_b", "all_mean_math_in_prompt",
        "all_mean_ethical_in_res_a", "all_mean_ethical_in_res_b", "all_std_len_log_res_b","all_std_len_log_res_a",
        "all_std_apologetic_words_res_a", "all_sum_question_mark_res_a","all_sum_ratio_res_a_prompt", "all_sum_ratio_res_b_prompt",
        'all_mean_count_words_prompt','all_mean_count_words_res_a', 'all_mean_count_words_res_b', 
        'all_mean_ratio_common_words_res_a', 'all_mean_ratio_common_words_res_b','all_mean_bold_in_res_a', 'all_mean_bold_in_res_b',
        'all_mean_list_points_res_a', 'all_mean_list_points_res_b', 'all_std_list_points_res_b','all_std_list_points_res_a'
    ]]


# PREPARE TRAIN DATAFRAME for top50
df_train = get_top_50_feats(train)

df_train_merge = pd.merge(train[['id','target']], df_train, on="id")

feats = list(df_train_merge.select_dtypes(["bool", "number"]).columns)
black_list = ["target", "id"]
feats = [x for x in feats if x not in black_list]

X_top50 = df_train_merge[feats].fillna(-1).values


# X, Y, MODEL
X = np.hstack([combined_w2v, X_top50]) # create X as W2V and TOP50
y = train["target"]

model = xgb.XGBClassifier(max_depth=8, n_estimators=70, learning_rate=0.1,
                          reg_lambda = 1.5, reg_alpha = 1.2, booster = 'gbtree', random_state=0)
model.fit(X, y)



# TRANSFORM TEST DATASET - WORD2VEC
run_eval_for_string(test, cols)

vec_columns = []

for col in cols_to_vec:
    vec_columns.append(np.array([document_vector(text) for text in test[col].map(flat_text_arr)]))
    
test_w2v = np.hstack(vec_columns)


# PREPARE TEST DATAFRAME for top50
df_test = get_top_50_feats(test)
X_test_top50 = df_test[feats].fillna(-1).values


# CREATE X_TEST as W2V and TOP50
X_test = np.hstack([test_w2v, X_test_top50]) 


# PREDICT
y_test = model.predict_proba(X_test)


# FILE TO SUBMIT
array_sub = np.concatenate((test["id"].values.reshape(-1,1), y_test), axis =1)
df_sub = pd.DataFrame(data=array_sub,  columns=['id','winner_model_a','winner_model_b', 'winner_tie']) 
df_sub['id'] = df_sub['id'].astype(int) 
df_sub.to_csv("submission.csv", index=False)