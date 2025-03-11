import numpy as np

class KeypointObject():
    """
    A class to encapsulate keypoint data and related information for training and testing.

    This class stores training and testing data, word dictionary, and decoder input/output data.
    It provides methods to access these data attributes.

    Attributes:
        __X_train (np.ndarray): Training data for the encoder.
        __Y_train (np.ndarray): Training labels for the encoder.
        __X_test (np.ndarray): Testing data for the encoder.
        __Y_test (np.ndarray): Testing labels for the encoder.
        __word_dict (dict[str, int]): Dictionary mapping words to their corresponding indices.
        __decoder_input_data_train (np.ndarray): Training data for the decoder input.
        __decoder_target_data_train (np.ndarray): Training data for the decoder target.
        __decoder_input_data_test (np.ndarray): Testing data for the decoder input.
        __decoder_target_data_test (np.ndarray): Testing data for the decoder target.

    Methods:
        get_train_data() -> tuple[np.ndarray, np.ndarray]:
            Returns the training data and labels for the encoder.
        
        get_test_data() -> tuple[np.ndarray, np.ndarray]:
            Returns the testing data and labels for the encoder.
        
        get_word_dict() -> dict[str, int]:
            Returns the word dictionary.
        
        get_decoder_train_data() -> tuple[np.ndarray, np.ndarray]:
            Returns the training data for the decoder input and target.
        
        get_decoder_test_data() -> tuple[np.ndarray, np.ndarray]:
            Returns the testing data for the decoder input and target.
    """

    def __init__(
            self,
            X_train: np.ndarray,
            Y_train: np.ndarray,
            X_test: np.ndarray,
            Y_test: np.ndarray,
            word_dict: dict[str, int],
            decoder_input_data_train: np.ndarray,
            decoder_target_data_train: np.ndarray,
            decoder_input_data_test: np.ndarray,
            decoder_target_data_test: np.ndarray,
    ):
        self.__X_train = X_train
        self.__Y_train = Y_train
        self.__X_test = X_test
        self.__Y_test = Y_test
        self.__word_dict = word_dict
        self.__decoder_input_data_train = decoder_input_data_train
        self.__decoder_target_data_train = decoder_target_data_train
        self.__decoder_input_data_test = decoder_input_data_test
        self.__decoder_target_data_test = decoder_target_data_test

    def get_train_data(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns the training data and labels for the encoder.

        Returns:
            tuple[np.ndarray, np.ndarray]: Training data and labels.
        """
        return self.__X_train, self.__Y_train
    
    def get_test_data(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns the testing data and labels for the encoder.

        Returns:
            tuple[np.ndarray, np.ndarray]: Testing data and labels.
        """
        return self.__X_test, self.__Y_test

    def get_word_dict(self) -> dict[str, int]:
        """
        Returns the word dictionary.

        Returns:
            dict[str, int]: Dictionary mapping words to their corresponding indices.
        """
        return self.__word_dict
    
    def get_decoder_train_data(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns the training data for the decoder input and target.

        Returns:
            tuple[np.ndarray, np.ndarray]: Decoder input and target training data.
        """
        return self.__decoder_input_data_train, self.__decoder_target_data_train
    
    def get_decoder_test_data(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns the testing data for the decoder input and target.

        Returns:
            tuple[np.ndarray, np.ndarray]: Decoder input and target testing data.
        """
        return self.__decoder_input_data_test, self.__decoder_target_data_test