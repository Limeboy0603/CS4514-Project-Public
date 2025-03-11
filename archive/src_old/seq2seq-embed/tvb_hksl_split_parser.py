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
    """

    def __init__(self, file: str):
        """
        Initializes the tvb_hksl_split_parser with the given CSV file.

        Args:
            file (str): The path to the CSV file containing training information.
        """
        self.file = file
        self.train_info = pd.read_csv(self.file, delimiter="|") 
        # extend the dataframe with extracted information
        self.train_info["glosses_tokenized"] = self.train_info["glosses"].str.split(' ')
        self.train_info["date"] = self.train_info["id"].str.split('/').apply(lambda x: x[0])
        self.train_info["frames"] = self.train_info["id"].str.split('/').apply(lambda x: x[1])
        self.train_info["length"] = self.train_info["frames"].str.split('-').apply(lambda x: int(x[1]) - int(x[0]) + 1)

    def get_train_id(self) -> pd.Series:
        """
        Returns the training IDs, with path separators adjusted for Windows if necessary.

        Returns:
            pd.Series: The training IDs.
        """
        if os.name == "nt":
            return self.train_info["id"].str.replace("/", "\\")
        return self.train_info["id"]

    def get_train_date(self) -> pd.Series:
        """
        Returns the dates extracted from the training IDs.

        Returns:
            pd.Series: The dates.
        """
        return self.train_info["date"]
    
    def get_train_frames(self) -> pd.Series:
        """
        Returns the frame ranges extracted from the training IDs.

        Returns:
            pd.Series: The frame ranges.
        """
        return self.train_info["frames"]

    def get_train_length(self) -> pd.Series:
        """
        Returns the lengths of the frame ranges.

        Returns:
            pd.Series: The lengths of the frame ranges.
        """
        return self.train_info["length"]

    def get_train_glosses_tokenized(self) -> pd.Series:
        """
        Returns the tokenized glosses.

        Returns:
            pd.Series: The tokenized glosses.
        """
        return self.train_info["glosses_tokenized"]