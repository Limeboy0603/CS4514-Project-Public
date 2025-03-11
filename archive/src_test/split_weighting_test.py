import pandas as pd
import os

class tvb_hksl_split_parser():
    """
    A class to parse and process the TVB-HKSL dataset split information.

    This class reads a CSV file containing training information and processes it to extract
    various details such as glosses, dates, frames, and lengths. It provides methods to access
    these processed details.

    Attributes:
        file (str): The path to the CSV file containing training information.
        train_info (pd.DataFrame): DataFrame containing the processed training information.

    Methods:
        get_train_id() -> pd.Series:
            Returns the training IDs, with path separators adjusted for Windows if necessary.
        
        get_train_date() -> pd.Series:
            Returns the dates extracted from the training IDs.
        
        get_train_frames() -> pd.Series:
            Returns the frame ranges extracted from the training IDs.
        
        get_train_length() -> pd.Series:
            Returns the lengths of the frame ranges.
        
        get_train_glosses_tokenized() -> pd.Series:
            Returns the tokenized glosses.

        get_max_train_glosses_length() -> int:
            Returns the maximum length of the tokenized glosses.
    """

    def __init__(self, file: str):
        self.file = file
        self.train_info = pd.read_csv(self.file, delimiter="|") 
        # extend the dataframe with extracted information
        self.train_info["glosses_tokenized"] = self.train_info["glosses"].str.split(' ')
        # self.train_info["date"] = self.train_info["id"].str.split('/').apply(lambda x: x[0])
        self.train_info["frames"] = self.train_info["id"].str.split('/').apply(lambda x: x[1])
        self.train_info["length"] = self.train_info["frames"].str.split('-').apply(lambda x: int(x[1]) - int(x[0]) + 1)

    def get_train_id(self) -> pd.Series:
        if os.name == "nt": # for windows system only
            return self.train_info["id"].str.replace("/", "\\")
        return self.train_info["id"]

    # def get_train_date(self) -> pd.Series:
    #     return self.train_info["date"]
    
    # def get_train_frames(self) -> pd.Series:
    #     return self.train_info["frames"]

    # def get_train_length(self) -> pd.Series:
    #     return self.train_info["length"]

    def get_train_glosses_tokenized(self) -> pd.Series:
        return self.train_info["glosses_tokenized"]

    def get_max_length(self) -> int:
        return self.train_info["length"].max()
    
if __name__ == "__main__":
    train_parser = tvb_hksl_split_parser("dataset/tvb-hksl-news/split/train.csv")
    test_parser = tvb_hksl_split_parser("dataset/tvb-hksl-news/split/test.csv")
    dev_parser = tvb_hksl_split_parser("dataset/tvb-hksl-news/split/dev.csv")

    train_tokens = train_parser.get_train_glosses_tokenized()
    test_tokens = test_parser.get_train_glosses_tokenized()
    dev_tokens = dev_parser.get_train_glosses_tokenized()

    # combine all tokens
    all_tokens = pd.concat([train_tokens, test_tokens, dev_tokens], ignore_index=True)

    # Convert tokenized sequences to strings for TF-IDF vectorizer
    all_documents = all_tokens.apply(lambda tokens: ' '.join(tokens))

    # Initialize TF-IDF Vectorizer
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_documents)

    # Get feature names (words)
    feature_names = vectorizer.get_feature_names_out()

    # Create a dictionary of TF-IDF scores for each word
    tfidf_scores = {}
    for doc_idx, doc in enumerate(tfidf_matrix):
        for word_idx, score in zip(doc.indices, doc.data):
            word = feature_names[word_idx]
            if word not in tfidf_scores:
                tfidf_scores[word] = []
            tfidf_scores[word].append(score)

    # Calculate the average TF-IDF score for each word
    avg_tfidf_scores = {word: sum(scores) / len(scores) for word, scores in tfidf_scores.items()}

    # Score each sequence based on the sum of TF-IDF scores of its words
    def score_sequence(sequence):
        return sum(avg_tfidf_scores.get(token, 0) for token in sequence)

    # Apply the scoring function to each sequence
    sequence_scores = all_tokens.apply(score_sequence)

    # export the scores to a csv file
    scores_df = pd.DataFrame()
    scores_df["sequence"] = all_documents
    scores_df["score"] = sequence_scores
    scores_df.to_csv("sequence_scores.csv", index=False)

    median_score = scores_df["score"].median()
    # duplicate samples with scores higher than the median, until the sample's score is less than the median
    high_score_samples = scores_df[scores_df["score"] > median_score]
    while not high_score_samples.empty:
        scores_df = pd.concat([scores_df, high_score_samples], ignore_index=True)
        high_score_samples = scores_df[scores_df["score"] > median_score]

    # export the duplicated samples to a csv file
    scores_df.to_csv("sequence_scores_duplicated.csv", index=False)