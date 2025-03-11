import pandas as pd

class tvb_hksl_split_parser():
    def __init__(self, file: str):
        self.file = file
        self.train_info = pd.read_csv(self.file, delimiter="|") 
        # extend the dataframe with extracted information
        self.train_info["glosses_tokenized"] = self.train_info["glosses"].str.split(' ')
        # self.train_info["date"] = self.train_info["id"].str.split('/').apply(lambda x: x[0])
        self.train_info["frames"] = self.train_info["id"].str.split('/').apply(lambda x: x[1])
        self.train_info["length"] = self.train_info["frames"].str.split('-').apply(lambda x: int(x[1]) - int(x[0]) + 1)
        # add <START> and <END> tokens to the glosses
        # self.train_info["glosses_tokenized"] = self.train_info["glosses_tokenized"].apply(lambda x: ["<START>"] + x + ["<END>"])
        self.train_info["glosses_length"] = self.train_info["glosses_tokenized"].apply(lambda x: len(x))
        

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

    def get_max_glosses_length(self) -> int:
        return self.train_info["glosses_length"].max()

    def pad_train_glosses_tokenized(self, max_length: int) -> pd.Series:
        self.train_info["glosses_tokenized"] = self.train_info["glosses_tokenized"].apply(lambda x: x + ["<END>"] * (max_length - len(x)))
        self.train_info["glosses_length"] = self.train_info["glosses_tokenized"].apply(lambda x: len(x))
        return self.train_info["glosses_tokenized"]
    
    def get_word_dict(self) -> dict:
        word_dict = {}
        for tokens in self.train_info["glosses_tokenized"]:
            for token in tokens:
                if token not in word_dict:
                    word_dict[token] = len(word_dict)
        return word_dict
    
train_parser = tvb_hksl_split_parser("../dataset/tvb-hksl-news/split/train.csv")
test_parser = tvb_hksl_split_parser("../dataset/tvb-hksl-news/split/test.csv")

train_gloss_list = train_parser.get_train_glosses_tokenized().to_list()
test_gloss_list = test_parser.get_train_glosses_tokenized().to_list()

print(train_gloss_list[0])
print(test_gloss_list[0])

relation_subsets = []
# relation_subsets is a list of dictionaries of
"""
{
    train: list of index in train_gloss_list,
    test: list of index in test_gloss_list,
    words: set of words in the subset
}
"""
