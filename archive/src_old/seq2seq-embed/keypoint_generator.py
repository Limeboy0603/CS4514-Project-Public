from tvb_hksl_split_parser import tvb_hksl_split_parser
import numpy as np
import os
from keras.preprocessing.sequence import pad_sequences # type: ignore
from alive_progress import alive_bar
from keypoint_object import KeypointObject
from config_object import ConfigObject
import json

class KeypointGenerator():
    """
    This class preprocesses the keypoints listed in the split file and returns the preprocessed keypoints as a KeypointObject.

    Attributes:
        train_parser (tvb_hksl_split_parser): The parser for the training data.
        test_parser (tvb_hksl_split_parser): The parser for the testing data.
        keypoint_dir (str): The directory where keypoints are stored.
        word_dict (dict[str, int]): The dictionary mapping words to their corresponding indices. This is primarily used for one-hot encoding.

    Methods:
        __pad_sequence_by_length(sequences: list, max_length: int, padding_value: str="") -> list:
            Pad the sequences by max_length.
        
        __generate_word_dict(sequences_list: list[list[str]]) -> dict[str, int]:
            Generate the word dictionary from the sequences list.
        
        __save_to_cache(data: np.ndarray, filename: str) -> np.ndarray:
            Save the data into cache. Returns the data as a memmap file if use_memmap is True, otherwise in normal ndarray format.
        
        __make_x(X_train: list[np.ndarray], X_test: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
            Turns the list of ndarray into ndarray.
        
        __make_y(Y_train: list[str], Y_test: list[str]) -> tuple[list[str], list[str]:
            Preprocess the Y_train and Y_test sequence data.
        
        __keypoints_reading() -> tuple[np.ndarray, list[str], np.ndarray, list[str]]:
            Read the keypoints from the split file via parser, then read the keypoints and return all X and Y data.
        
        __map_words_to_integers(Y_train: list[str], Y_test: list[str]) -> tuple[np.ndarray, np.ndarray]:
            Map the words to integers using the word dictionary, and convert the glosses to one-hot encoded vectors.
        
        __create_decoder_data(Y_train: np.ndarray, Y_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            Creates the decoder input and target data for the training and testing data.
        
        __preprocess_keypoints() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, int], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            Preprocess the keypoints regardless of type.
        
        get_provided_keypoints() -> KeypointObject:
            Preprocess the provided keypoints and returns the wrapped KeypointObject.
        
        get_mediapipe_keypoints_index() -> list[int]:
            Returns the indices of the keypoints that we want to keep for keypoints generated using Mediapipe.
        
        get_mediapipe_keypoints() -> KeypointObject:
            Preprocess the keypoints from the Mediapipe model, and returns the wrapped KeypointObject.
        
        get_cache_keypoints() -> KeypointObject:
            Retrieve the keypoints from cache and construct a KeypointObject.
            
        __store_word_dict() -> None:
            Store the word dictionary in cache as a JSON file.
        
        __have_cache() -> bool:
            Check if all cached files exist.
        
        get_cache_keypoints() -> KeypointObject:
            Retrieve the keypoints from cache and construct a KeypointObject.
    """
    def __init__(self, train_parser: tvb_hksl_split_parser, test_parser: tvb_hksl_split_parser, keypoint_dir: str, cache_dir: str, use_memmap: bool):
        self.train_parser = train_parser
        self.test_parser = test_parser
        self.keypoint_dir = keypoint_dir
        self.cache_dir = cache_dir
        self.use_memmap = use_memmap

        self.word_dict: dict[str, int] = {}
        
        # create a private keypoint index list for whenever X only needs specific features
        self.__x_feature_list = None

    @staticmethod
    def __pad_sequence_by_length(sequences: list, max_length: int, padding_value: str="") -> list:
        """
        Pad the sequences by max_length.

        Args:
            sequences (list): List of sequences to pad.
            max_length (int): The maximum length to pad to.
            padding_value (str): The value to pad the sequences with.

        Returns:
            list: A list of padded sequences 
        """
        padded_sequences = [sequence + [padding_value] * (max_length - len(sequence)) for sequence in sequences]
        return padded_sequences

    @staticmethod
    def __generate_word_dict(sequences_list: list[list[str]]) -> dict[str, int]:
        """
        Generate the word dictionary from the sequences list. The word dictionary contains the words and their corresponding integers.
        Some special tokens are inserted at the front of the dictionary.
        The dictionary is then used to map the words to integers.

        Args:
            sequences_list (list[list[list[str]]]): List of sequences to generate the word dictionary from.

        Returns:
            dict[str, int]: A dictionary containing the words and their corresponding integers.
        """
        print("Generating word dictionary...")
        word_dict = {}

        # special values
        word_dict["<PAD>"] = len(word_dict)
        word_dict["<START>"] = len(word_dict)
        word_dict["<END>"] = len(word_dict)
        word_dict["<X>"] = len(word_dict)
        word_dict["<BAD>"] = len(word_dict)
        word_dict["<MUMBLE>"] = len(word_dict)

        # normal glosses
        for sequences in sequences_list:
            for sequence in sequences:
                for word in sequence:
                    if word not in word_dict:
                        word_dict[word] = len(word_dict)
        return word_dict

    def __save_to_cache(self, data: np.ndarray, filename: str, dtype: str = "float32") -> np.ndarray:
        """
        Save the data into cache. The code will always store keypoints in the cache directory regardless of use_memmap.
        If use_memmap is True, reload the data as a memmap file and return it. This can hopefully reduce memory usage as keypoints are stored on disk rather than in memory.
        Otherwise, return the data as is.

        Args:
            data (np.ndarray): The data to save.
            filename (str): The filename to save the data to.
        
        Returns:
            np.ndarray: The data as a memmap file if use_memmap is True, otherwise in normal ndarray format.
        """
        assert filename.endswith(".npy"), "Filename must end with .npy"
        # np.save(os.path.join(self.cache_dir, filename), data)
        data_shape = data.shape
        print("Saving data to cache:", filename, data_shape)
        print(self.use_memmap)
        if self.use_memmap:
            if dtype == "int8":
                assert np.all(data >= 0), "Original Data contains negative values"
            memmap_file = np.memmap(os.path.join(self.cache_dir, filename), dtype=dtype, mode='w+', shape=data_shape)
            memmap_file[:] = data[:]
            if dtype == "int8":
                assert np.all(memmap_file >= 0), "Memmap Data contains negative values"
            memmap_file.flush()
            del data
            del memmap_file
            data = np.memmap(os.path.join(self.cache_dir, filename), dtype=dtype, mode='r', shape=data_shape)
            # if the type is int8, assert that all values are non-negative
            if dtype == "int8":
                assert np.all(data >= 0), "Data contains negative values"
        else:
            np.save(os.path.join(self.cache_dir, filename), data)
        return data
    
    def __make_x(self, X_train: list[np.ndarray], X_test: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """
        Makes the X_train and X_test numpy files.

        Args:
            X_train (list[]): Training data
            X_test (list): Testing data

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing the memmaped X_train and X_test.
        """
        # pad the sequences by the maximum length
        max_length = max([len(sequence) for sequence in X_train + X_test])
        X_train = pad_sequences(X_train, maxlen=max_length, dtype='float32', padding='post', value=0)
        X_test = pad_sequences(X_test, maxlen=max_length, dtype='float32', padding='post', value=0)

        X_train = self.__save_to_cache(X_train, "X_train.npy")
        X_test = self.__save_to_cache(X_test, "X_test.npy")

        return X_train, X_test

    def __make_y(self, Y_train: list[str], Y_test: list[str]) -> tuple[list[str], list[str]]:
        """
        Makes the Y_train and Y_test sequence data. Adds "<START>" and "<END>" to the start and end of each sequence, and pads the sequences by the maximum length.

        Args:
            Y_train (list): Training glosses
            Y_test (list): Testing glosses

        Returns:
            tuple[list[str], list[str]]: A tuple containing the Y_train and Y_test.
        """
        # Add "<START>" and "<END>" to the start and end of each sequence
        for i in range(len(Y_train)):
            Y_train[i] = ["<START>"] + Y_train[i] + ["<END>"]
        for i in range(len(Y_test)):
            Y_test[i] = ["<START>"] + Y_test[i] + ["<END>"]

        # if Y_train and Y_test have different lengths, pad the shorter one with "<PAD>"
        # ideally, the model should recognize np.zeros in X as padding values
        max_length = max([len(sequence) for sequence in Y_train + Y_test])
        Y_train = self.__pad_sequence_by_length(Y_train, max_length, padding_value="<PAD>")
        Y_test = self.__pad_sequence_by_length(Y_test, max_length, padding_value="<PAD>")

        # DEBUG: print the first 5 sequences of Y_train and Y_test
        # print("Y_train:", Y_train[:5])
        # print("Y_test:", Y_test[:5])

        return Y_train, Y_test


    def __keypoints_reading(self) -> tuple[np.ndarray, list[str], np.ndarray, list[str]]:
        """
        Read the keypoints from the split file via parser, then read the keypoints and return all X and Y data for training and tesing.

        Returns:
            tuple[np.ndarray, list[str], np.ndarray, list[str]]: A tuple containing the keypoints.

        Returned tuple elements:
        - X_train: Training data
        - Y_train: Training glosses
        - X_test: Testing data
        - Y_test: Testing glosses
        """
        train_id = self.train_parser.get_train_id()
        train_glosses_tokenized = self.train_parser.get_train_glosses_tokenized()
        train_frames = self.train_parser.get_train_frames()
        train_date = self.train_parser.get_train_date()
        train_length = self.train_parser.get_train_length()
        print(train_id, train_glosses_tokenized, train_frames, train_date, train_length)

        test_id = self.test_parser.get_train_id()
        test_glosses_tokenized = self.test_parser.get_train_glosses_tokenized()
        test_frames = self.test_parser.get_train_frames()
        test_date = self.test_parser.get_train_date()
        test_length = self.test_parser.get_train_length()
        print(test_id, test_glosses_tokenized, test_frames, test_date, test_length)

        def apply_keypoint_filter(keypoint: np.ndarray, keypoint_index: list[int]) -> np.ndarray:
            return keypoint[:, keypoint_index]

        # Step 2: Preprocess the data
        # for each id, we need to load the corresponding keypoint file in numpy format
        # concatenate the keypoint files to form the input tensor
        # output tensor is the glosses tokenized
        print("Reading training keypoints...")
        X_train = []
        Y_train = []
        with alive_bar(len(train_date)) as bar:
            for i in range(len(train_date)):
                # keypoint_file = f"{keypoint_dir}/{train_id[i]}.npy"
                keypoint_file = os.path.join(self.keypoint_dir, f"{train_id[i]}.npy")
                keypoint = np.load(keypoint_file)
                # for 3D, we need to remove the z coordinate
                if len(keypoint.shape) == 3:
                    keypoint = keypoint[:, :, :2]
                keypoint = apply_keypoint_filter(keypoint, self.__x_feature_list)
                keypoint = keypoint.reshape(keypoint.shape[0], -1)
                X_train.append(keypoint)
                Y_train.append(train_glosses_tokenized[i])
                bar()
        
        print("Reading testing keypoints...")
        X_test = []
        Y_test = []
        with alive_bar(len(test_date)) as bar:
            for i in range(len(test_date)):
                # keypoint_file = f"{keypoint_dir}/{test_id[i]}.npy"
                keypoint_file = os.path.join(self.keypoint_dir, f"{test_id[i]}.npy")
                keypoint = np.load(keypoint_file)
                # for 3D, we need to remove the z coordinate
                if len(keypoint.shape) == 3:
                    keypoint = keypoint[:, :, :2]
                keypoint = apply_keypoint_filter(keypoint, self.__x_feature_list)
                keypoint = keypoint.reshape(keypoint.shape[0], -1)
                X_test.append(keypoint)
                Y_test.append(test_glosses_tokenized[i])
                bar()

        self.__x_feature_list = None

        X_train, X_test = self.__make_x(X_train, X_test)
        Y_train, Y_test = self.__make_y(Y_train, Y_test)

        return X_train, Y_train, X_test, Y_test

    def __map_words_to_integers(self, Y_train: list[str], Y_test: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """
        Map the words to integers using the word dictionary, and convert the glosses to one-hot encoded vectors.

        Args:
            Y_train (list): Training glosses
            Y_test (list): Testing glosses

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing the mapped words to integers.

        Returned tuple elements:
        - Y_train: Training glosses that are one-hot encoded
        - Y_test: Testing glosses that are one-hot encoded
        """
        print("Mapping words to integers...")
        for i in range(len(Y_train)):
            Y_train[i] = [self.word_dict[word] for word in Y_train[i]]
        for i in range(len(Y_test)):
            Y_test[i] = [self.word_dict[word] for word in Y_test[i]]

        # generate one-hot encoded vectors
        # Y_train_one_hot = np.zeros((len(Y_train), len(Y_train[0]), len(self.word_dict)))
        # Y_test_one_hot = np.zeros((len(Y_test), len(Y_test[0]), len(self.word_dict)))

        # for i in range(len(Y_train)):
        #     for j in range(len(Y_train[i])):
        #         Y_train_one_hot[i, j, Y_train[i][j]] = 1
        
        # for i in range(len(Y_test)):
        #     for j in range(len(Y_test[i])):
        #         Y_test_one_hot[i, j, Y_test[i][j]] = 1

        # Y_train_one_hot = self.__save_to_cache(Y_train_one_hot, "Y_train.npy", dtype="int8")
        # Y_test_one_hot = self.__save_to_cache(Y_test_one_hot, "Y_test.npy", dtype="int8")

        # return Y_train_one_hot, Y_test_one_hot

        # turn Y_train and Y_test into numpy arrays
        Y_train = np.array(Y_train)
        Y_test = np.array(Y_test)

        assert np.all(Y_train >= 0), "Y_train contains negative values"
        assert np.all(Y_test >= 0), "Y_test contains negative values"

        Y_train = self.__save_to_cache(Y_train, "Y_train.npy", dtype="int8")
        Y_test = self.__save_to_cache(Y_test, "Y_test.npy", dtype="int8")

        assert np.all(Y_train >= 0), "Y_train contains negative values after memmap"
        assert np.all(Y_test >= 0), "Y_test contains negative values after memmap"

        return Y_train, Y_test
    
    def __create_decoder_data(self, Y_train: np.ndarray, Y_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Creates the decoder data for the training and testing data.

        Args:
            Y_train (np.ndarray): Training glosses
            Y_test (np.ndarray): Testing glosses

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing the decoder input and target data for the training and testing data.

        Returned tuple elements:
        - decoder_input_data_train: Training data for the decoder input
        - decoder_target_data_train: Training data for the decoder target
        - decoder_input_data_test: Testing data for the decoder input
        - decoder_target_data_test: Testing data for the decoder target
        """
        # TODO: Rewrite this part
        
        # DEBUG: print Y train and Y test
        # print("Y_train:", Y_train)
        # print("Y_test:", Y_test)

        # CHECK: whether all values of Y_train and Y_test are within the range of the word dictionary
        assert np.all(Y_train < len(self.word_dict)), "Y_train contains values that are not in the word dictionary"
        assert np.all(Y_test < len(self.word_dict)), "Y_test contains values that are not in the word dictionary"

        # CHECK: whether all values of Y_train and Y_test are non-negative
        assert np.all(Y_train >= 0), "Y_train contains negative values"
        assert np.all(Y_test >= 0), "Y_test contains negative values"

        decoder_input_data_train = np.copy(Y_train)
        decoder_target_data_train = np.copy(Y_train)
        decoder_input_data_test = np.copy(Y_test)
        decoder_target_data_test = np.copy(Y_test)

        # shift the target data by one
        decoder_target_data_train = np.roll(decoder_target_data_train, -1, axis=1)
        decoder_target_data_test = np.roll(decoder_target_data_test, -1, axis=1)

        # set the last element of target data to that which represent a padding value
        # one-hot
        # decoder_target_data_train[:, -1, :] = 0
        # decoder_target_data_test[:, -1, :] = 0
        # decoder_target_data_train[:, -1, self.word_dict["<PAD>"]] = 1
        # decoder_target_data_test[:, -1, self.word_dict["<PAD>"]] = 1
        # normal
        decoder_target_data_train[:, -1] = 0
        decoder_target_data_test[:, -1] = 0

        decoder_input_data_train = self.__save_to_cache(decoder_input_data_train, "decoder_input_data_train.npy", dtype="int8")
        decoder_target_data_train = self.__save_to_cache(decoder_target_data_train, "decoder_target_data_train.npy", dtype="int8")
        decoder_input_data_test = self.__save_to_cache(decoder_input_data_test, "decoder_input_data_test.npy", dtype="int8")
        decoder_target_data_test = self.__save_to_cache(decoder_target_data_test, "decoder_target_data_test.npy", dtype="int8")

        return decoder_input_data_train, decoder_target_data_train, decoder_input_data_test, decoder_target_data_test

    def __preprocess_keypoints(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, int], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Preprocess the keypoints regardless of type.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, int], np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing the preprocessed keypoints
        
        Returned tuple elements:
        - X_train: Training data
        - Y_train: Training glosses
        - X_test: Testing data
        - Y_test: Testing glosses
        - word_dict: Word dictionary
        - decoder_input_data_train: Training data for the decoder input
        - decoder_target_data_train: Training data for the decoder target
        - decoder_input_data_test: Testing data for the decoder input
        - decoder_target_data_test: Testing data for the decoder target
        """
        print("Starting keypoints preprocessing...")
        X_train, Y_train, X_test, Y_test = self.__keypoints_reading()
        self.word_dict = self.__generate_word_dict([Y_train, Y_test])
        Y_train, Y_test = self.__map_words_to_integers(Y_train, Y_test)
        decoder_input_data_train, decoder_target_data_train, decoder_input_data_test, decoder_target_data_test = self.__create_decoder_data(Y_train, Y_test)

        with open(os.path.join(self.cache_dir, "word_dict.json"), 'w+') as f:
            json.dump(self.word_dict, f)

        shapes = [
            X_train.shape,
            Y_train.shape,
            X_test.shape,
            Y_test.shape,
            decoder_input_data_train.shape,
            decoder_target_data_train.shape,
            decoder_input_data_test.shape,
            decoder_target_data_test.shape
        ]

        with open(os.path.join(self.cache_dir, "shapes.json"), 'w+') as f:
            json.dump(shapes, f)

        print("Keypoints preprocessing completed.")
        return X_train, Y_train, X_test, Y_test, self.word_dict, decoder_input_data_train, decoder_target_data_train, decoder_input_data_test, decoder_target_data_test

    def get_provided_keypoints(self) -> KeypointObject:
        """
        Preprocess the provided keypoints and returns the wrapped KeypointObject.

        Returns:
            KeypointObject: An object containing the preprocessed keypoints.
        """

        X_train, Y_train, X_test, Y_test, word_dict, decoder_input_data_train, decoder_target_data_train, decoder_input_data_test, decoder_target_data_test = self.__preprocess_keypoints()
        print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape, sep="\n")
        """
        For provide keypoints:

        dev:
        (322, 376, 266)
        (322, 45, 1556)
        (322, 376, 266)
        (322, 45, 1556)
        train:
        (6516, 376, 266)
        (6516, 47, 6516)
        (322, 376, 266)
        (322, 47, 6516)

        X: (
            number of samples, 
            number of frames, (highest number of frames among all samples in the dataset)
            number of features (coordinates of keypoints, flattened into 1D)
        )
        Y: (
            number of samples,
            number of words in the glosses,
            number of class labels (one-hot encoded)
        )
        """
        print("Total number of words:", len(word_dict)) # 1554

        # DEBUG: count the number of negative values in X_train and X_test
        print("Number of negative values in X_train:", np.sum(X_train < 0)) # 13728
        print("Number of negative values in X_test:", np.sum(X_test < 0)) # 496

        # DEBUG: print the largest and smallest possible value in X_train and X_test
        print("Largest value in X_train", np.max(X_train)) # dev: 137294, train: 1297826
        print("Smallest value in X_train", np.min(X_train)) # dev: -102974 train: -1842615
        print("Largest value in X_test", np.max(X_test)) # 112651
        print("Smallest value in X_test", np.min(X_test)) # -397718

        return KeypointObject(X_train, Y_train, X_test, Y_test, word_dict, decoder_input_data_train, decoder_target_data_train, decoder_input_data_test, decoder_target_data_test)

    # this part may be reused for demo purposes, hence set as public static
    @staticmethod
    def get_mediapipe_keypoints_index() -> list[int]:
        """
        Returns the indices of the keypoints that we want to keep.
        
        For the third dimension, we only want to keep the coordinates of
        - Pose
        - Face border
        - Lips
        - Eyes
        - Eyebrows
        - Nose
        - Face Oval (border of face)
        - Left hand
        - Right hand

        This is because we want the keypoints to be robust, thus the facial features that are unique to each signer are discarded.

        Reference: https://github.com/LearningnRunning/py_face_landmark_helper/blob/main/mediapipe_helper/config.py
        Image: https://raw.githubusercontent.com/google/mediapipe/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
        Related stack overflow post: https://stackoverflow.com/questions/74901522/can-mediapipe-specify-which-parts-of-the-face-mesh-are-the-lips-or-nose-or-eyes

        Returns:
            list[int]: List of indices of the keypoints to keep.
        """
        # pose with visibility
        # POSE = list(range(0, 33*4))

        # pose without visibility
        POSE_UNPROCESSED = range(0, 33*4)
        POSE = [i for i in POSE_UNPROCESSED if i % 4 != 3]
        # for x, y only
        # POSE = [i for i in POSE_UNPROCESSED if i % 4 != 2 and i % 4 != 3]

        # face
        # NOTE: the following keypoint indices are HARD-CODED based on the visualization of the face mesh
        # reference: https://github.com/LearningnRunning/py_face_landmark_helper/blob/main/mediapipe_helper/config.py
        # image: https://raw.githubusercontent.com/google/mediapipe/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
        # related stack overflow post: https://stackoverflow.com/questions/74901522/can-mediapipe-specify-which-parts-of-the-face-mesh-are-the-lips-or-nose-or-eyes
        FACE_LIPS = [0, 267, 269, 270, 13, 14, 17, 402, 146, 405, 409, 415, 291, 37, 39, 40, 178, 308, 181, 310, 311, 312, 185, 314, 317, 318, 61, 191, 321, 324, 78, 80, 81, 82, 84, 87, 88, 91, 95, 375]
        LEFT_EYE = [384, 385, 386, 387, 388, 390, 263, 362, 398, 466, 373, 374, 249, 380, 381, 382]
        LEFT_EYEBROW = [293, 295, 296, 300, 334, 336, 276, 282, 283, 285]
        RIGHT_EYE = [160, 33, 161, 163, 133, 7, 173, 144, 145, 246, 153, 154, 155, 157, 158, 159]
        RIGHT_EYEBROW = [65, 66, 70, 105, 107, 46, 52, 53, 55, 63]
        FACE_NOSE = [1, 2, 4, 5, 6, 19, 275, 278, 294, 168, 45, 48, 440, 64, 195, 197, 326, 327, 344, 220, 94, 97, 98, 115]
        FACE_OVAL = [132, 389, 136, 10, 397, 400, 148, 149, 150, 21, 152, 284, 288, 162, 297, 172, 176, 54, 58, 323, 67, 454, 332, 338, 93, 356, 103, 361, 234, 109, 365, 379, 377, 378, 251, 127]
        FACE_UNPROCESSED = [item + 33*4 for sublist in [FACE_LIPS, LEFT_EYE, LEFT_EYEBROW, RIGHT_EYE, RIGHT_EYEBROW, FACE_NOSE, FACE_OVAL] for item in sublist]
        # face keypoints are in x, y, z format flattened, so we need to capture all x, y, z values
        FACE = [i for j in range(0, len(FACE_UNPROCESSED), 3) for i in range(FACE_UNPROCESSED[j], FACE_UNPROCESSED[j] + 3)]
        # for x, y only
        # FACE = [i for j in range(0, len(FACE_UNPROCESSED), 3) for i in range(FACE_UNPROCESSED[j], FACE_UNPROCESSED[j] + 2)]

        # hands
        LEFT_HAND = list(range(33*4 + 468*3, 33*4 + 468*3 + 21*3))
        RIGHT_HAND = list(range(33*4 + 468*3 + 21*3, 33*4 + 468*3 + 21*3 + 21*3))
        # for x, y only
        # LEFT_HAND = [i for i in list(range(33*4 + 468*3, 33*4 + 468*3 + 21*3)) if i % 3 != 2]
        # RIGHT_HAND = [i for i in list(range(33*4 + 468*3 + 21*3, 33*4 + 468*3 + 21*3 + 21*3)) if i % 3 != 2]
        KEYPOINTS_INDEX = POSE + FACE + LEFT_HAND + RIGHT_HAND
        return KEYPOINTS_INDEX

    def get_mediapipe_keypoints(self) -> KeypointObject:
        """
        Preprocess the keypoints from the Mediapipe model, and return the wrapped KeypointObject.

        Returns:
            KeypointObject: An object containing the preprocessed keypoints.
        """
        self.__x_feature_list = self.get_mediapipe_keypoints_index()
        X_train, Y_train, X_test, Y_test, word_dict, decoder_input_data_train, decoder_target_data_train, decoder_input_data_test, decoder_target_data_test = self.__preprocess_keypoints()

        return KeypointObject(X_train, Y_train, X_test, Y_test, word_dict, decoder_input_data_train, decoder_target_data_train, decoder_input_data_test, decoder_target_data_test)

    def get_cache_keypoints(self) -> KeypointObject:
        # return None # uncomment this line after testing
        """
        Retrieve the keypoints from cache and construct a KeypointObject.
        If memmap is enabled, load the data as memmap files. Otherwise, read it as normal numpy arrays.

        Returns:
            KeypointObject: An object containing the preprocessed keypoints.
        """
        with open(f"{self.cache_dir}/word_dict.json", "r") as f:
            word_dict = json.load(f)

        if self.use_memmap:
            # parse the shapes
            with open(f"{self.cache_dir}/shapes.json", "r") as f:
                shapes = json.load(f)
            
            for i, shape in enumerate(shapes):
                shapes[i] = tuple(shape)

            X_train = np.memmap(f"{self.cache_dir}/X_train.npy", dtype='float32', mode='r', shape=shapes[0])
            Y_train = np.memmap(f"{self.cache_dir}/Y_train.npy", dtype='int8', mode='r', shape=shapes[1])
            X_test = np.memmap(f"{self.cache_dir}/X_test.npy", dtype='float32', mode='r', shape=shapes[2])
            Y_test = np.memmap(f"{self.cache_dir}/Y_test.npy", dtype='int8', mode='r', shape=shapes[3])
            decoder_input_data_train = np.memmap(f"{self.cache_dir}/decoder_input_data_train.npy", dtype='int8', mode='r', shape=shapes[4])
            decoder_target_data_train = np.memmap(f"{self.cache_dir}/decoder_target_data_train.npy", dtype='int8', mode='r', shape=shapes[5])
            decoder_input_data_test = np.memmap(f"{self.cache_dir}/decoder_input_data_test.npy", dtype='int8', mode='r', shape=shapes[6])
            decoder_target_data_test = np.memmap(f"{self.cache_dir}/decoder_target_data_test.npy", dtype='int8', mode='r', shape=shapes[7])
        else:
            X_train = np.load(f"{self.cache_dir}/X_train.npy")
            Y_train = np.load(f"{self.cache_dir}/Y_train.npy")
            X_test = np.load(f"{self.cache_dir}/X_test.npy")
            Y_test = np.load(f"{self.cache_dir}/Y_test.npy")
            decoder_input_data_train = np.load(f"{self.cache_dir}/decoder_input_data_train.npy")
            decoder_target_data_train = np.load(f"{self.cache_dir}/decoder_target_data_train.npy")
            decoder_input_data_test = np.load(f"{self.cache_dir}/decoder_input_data_test.npy")
            decoder_target_data_test = np.load(f"{self.cache_dir}/decoder_target_data_test.npy")
        print("All keypoints have been retrieved.")

        return KeypointObject(X_train, Y_train, X_test, Y_test, word_dict, decoder_input_data_train, decoder_target_data_train, decoder_input_data_test, decoder_target_data_test)

    def __store_word_dict(self) -> None:
        """
        Store the word dictionary in cache.
        """
        with open(f"{self.cache_dir}/word_dict.json", "w+") as f:
            json.dump(self.word_dict, f, indent=4)

    def __have_cache(self) -> bool:
        """
        Check if all cached files exist.
        
        Returns:
            bool: True if all cached files exist, False otherwise
        """
        required_files = [
            "X_train.npy",
            "Y_train.npy",
            "X_test.npy",
            "Y_test.npy",
            "word_dict.json",
            "decoder_input_data_train.npy",
            "decoder_target_data_train.npy",
            "decoder_input_data_test.npy",
            "decoder_target_data_test.npy",
            "shapes.json",
        ]

        if all(os.path.exists(f"{self.cache_dir}/{file}") for file in required_files):
            return True
    
        # print the missing files
        print("Missing files:")
        for file in required_files:
            if not os.path.exists(f"{self.cache_dir}/{file}"):
                print(f"{file} is missing.")
        return False
    

    def get_keypoints(self, keypoint_type: str) -> KeypointObject:
        """
        Get the keypoints based on the keypoint type.
        If the keypoints are already stored in cache, load the keypoints from cache to save time.
        Otherwise, preprocess the keypoints and return the keypoints as a KeypointObject, and store the keypoints in cache while preprocessing.

        Args:
            keypoint_type (str): The type of keypoints to get.

        Returns:
            KeypointObject: An object containing the preprocessed keypoints.
        """
        if self.__have_cache():
            print("Full cache directory exists.")
            return self.get_cache_keypoints()
        else:
            os.makedirs(self.cache_dir, exist_ok=True)

        if keypoint_type == "provided":
            kp = self.get_provided_keypoints()
        elif keypoint_type == "mediapipe":
            kp = self.get_mediapipe_keypoints()
        else:
            raise ValueError("Invalid keypoint type")
        
        self.__store_word_dict()
        return kp

# run the main function to pregenerate all keypoints in cache
# otherwise, the keypoints will be generated on-the-fly
if __name__ == "__main__":
    output_paths = [
        "keypoint_output/provided_dev",
        "keypoint_output/mediapipe_dev",
        "keypoint_output/provided_train",
        "keypoint_output/mediapipe_train",
    ]

    for output_path in output_paths:
        os.makedirs(output_path, exist_ok=True)

    for config_path in [
                        "config/wsl/provided_dev.yaml",
                        "config/wsl/mediapipe_dev.yaml",
                        "config/wsl/provided_train.yaml",
                        "config/wsl/mediapipe_train.yaml",
    ]:
        config_module = ConfigObject(config_path)
        word_dict_path = config_module.get_word_dict_path()
        train_file = config_module.get_train_file()
        test_file = config_module.get_test_file()
        keypoint_dir = config_module.get_keypoint_dir()
        cache_dir = config_module.get_cache_dir()

        train_parser = tvb_hksl_split_parser(train_file)
        test_parser = tvb_hksl_split_parser(test_file)

        keypoint_generator = KeypointGenerator(train_parser, test_parser, keypoint_dir, cache_dir, True)
        keypoint_obj = keypoint_generator.get_keypoints(config_module.get_type())

        # save the first sample of X_train, Y_train, X_test, Y_test, decoder_input_data_train, decoder_target_data_train, decoder_input_data_test, decoder_target_data_test as txt
        X_train = keypoint_obj.X_train[0]
        Y_train = keypoint_obj.Y_train[0]
        X_test = keypoint_obj.X_test[0]
        Y_test = keypoint_obj.Y_test[0]
        decoder_input_data_train = keypoint_obj.decoder_input_data_train[0]
        decoder_target_data_train = keypoint_obj.decoder_target_data_train[0]
        decoder_input_data_test = keypoint_obj.decoder_input_data_test[0]
        decoder_target_data_test = keypoint_obj.decoder_target_data_test[0]

        np.savetxt(f"{output_path}/X_train.txt", X_train)
        np.savetxt(f"{output_path}/Y_train.txt", Y_train)
        np.savetxt(f"{output_path}/X_test.txt", X_test)
        np.savetxt(f"{output_path}/Y_test.txt", Y_test)
        np.savetxt(f"{output_path}/decoder_input_data_train.txt", decoder_input_data_train)
        np.savetxt(f"{output_path}/decoder_target_data_train.txt", decoder_target_data_train)
        np.savetxt(f"{output_path}/decoder_input_data_test.txt", decoder_input_data_test)
        np.savetxt(f"{output_path}/decoder_target_data_test.txt", decoder_target_data_test)