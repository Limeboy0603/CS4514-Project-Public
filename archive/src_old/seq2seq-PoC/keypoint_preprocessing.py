from tvb_hksl_split_parser import tvb_hksl_split_parser
import numpy as np
import os
from keras.preprocessing.sequence import pad_sequences # type: ignore
from alive_progress import alive_bar

def pad_sequence_by_length(sequences, max_length: int, padding_value: str=""):
    padded_sequences = [sequence + [padding_value] * (max_length - len(sequence)) for sequence in sequences]
    return padded_sequences

def generate_word_dict(sequences_list):
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

def __keypoints_reading(train_parser: tvb_hksl_split_parser, test_parser: tvb_hksl_split_parser, keypoint_dir: str):
    train_id = train_parser.get_train_id()
    train_glosses_tokenized = train_parser.get_train_glosses_tokenized()
    train_frames = train_parser.get_train_frames()
    train_date = train_parser.get_train_date()
    train_length = train_parser.get_train_length()
    print(train_id, train_glosses_tokenized, train_frames, train_date, train_length)

    test_id = test_parser.get_train_id()
    test_glosses_tokenized = test_parser.get_train_glosses_tokenized()
    test_frames = test_parser.get_train_frames()
    test_date = test_parser.get_train_date()
    test_length = test_parser.get_train_length()
    print(test_id, test_glosses_tokenized, test_frames, test_date, test_length)

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
            keypoint_file = os.path.join(keypoint_dir, f"{train_id[i]}.npy")
            keypoint = np.load(keypoint_file)
            X_train.append(keypoint)
            Y_train.append(train_glosses_tokenized[i])
            bar()

    print("Reading testing keypoints...")
    X_test = []
    Y_test = []
    with alive_bar(len(test_date)) as bar:
        for i in range(len(test_date)):
            # keypoint_file = f"{keypoint_dir}/{test_id[i]}.npy"
            keypoint_file = os.path.join(keypoint_dir, f"{test_id[i]}.npy")
            keypoint = np.load(keypoint_file)
            X_test.append(keypoint)
            Y_test.append(test_glosses_tokenized[i])
            bar()

    X_test = pad_sequences(X_test, padding="post")
    # Y_test = pad_sequences(Y_test, padding="post")
    X_train = pad_sequences(X_train, padding="post")
    # Y_train = pad_sequences(Y_train, padding="post")
    # Y_test = pad_sequence_with_strings(Y_test, padding_value="<PAD>")
    # Y_train = pad_sequence_with_strings(Y_train, padding_value="<PAD>")

    # Add "<START>" and "<END>" to the start and end of each sequence
    for i in range(len(Y_train)):
        Y_train[i] = ["<START>"] + Y_train[i] + ["<END>"]
    for i in range(len(Y_test)):
        Y_test[i] = ["<START>"] + Y_test[i] + ["<END>"]
    # if Y_train and Y_test have different lengths, pad the shorter one with "<PAD>"
    # ideally, the model should recognize np.zeros in X as padding values
    max_length = max([len(sequence) for sequence in Y_train + Y_test])
    Y_train = pad_sequence_by_length(Y_train, max_length, padding_value="<PAD>")
    Y_test = pad_sequence_by_length(Y_test, max_length, padding_value="<PAD>")


    # if X_train.shape[1] != X_test.shape[1], pad the shorter one with zeros
    max_length = max(X_train.shape[1], X_test.shape[1])
    if X_train.shape[1] != X_test.shape[1]:
        X_train = pad_sequences(X_train, maxlen=max_length, padding="post")
        X_test = pad_sequences(X_test, maxlen=max_length, padding="post")
    elif X_train.shape[1] == X_test.shape[1]:
        X_train = pad_sequences(X_train, maxlen=max_length, padding="post")
        X_test = pad_sequences(X_test, maxlen=max_length, padding="post")
    
    return X_train, Y_train, X_test, Y_test

def __map_words_to_integers(Y_train, Y_test, word_dict):
    print("Mapping words to integers...")
    for i in range(len(Y_train)):
        Y_train[i] = [word_dict[word] for word in Y_train[i]]
    for i in range(len(Y_test)):
        Y_test[i] = [word_dict[word] for word in Y_test[i]]
    Y_train = np.array([np.eye(len(word_dict))[sequence] for sequence in Y_train])
    Y_test = np.array([np.eye(len(word_dict))[sequence] for sequence in Y_test])
    return Y_train, Y_test

def __convert_to_numpy_array(X_train, Y_train, X_test, Y_test):
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)
    return X_train, Y_train, X_test, Y_test

# input: nparray, nparray, dict[str, int]
def __create_decoder_data(Y_train, Y_test):
    decoder_input_data_train = np.zeros_like(Y_train)
    decoder_target_data_train = np.zeros_like(Y_train)
    decoder_input_data_test = np.zeros_like(Y_test)
    decoder_target_data_test = np.zeros_like(Y_test)

    # shifting
    for i in range(len(Y_train)):
        decoder_input_data_train[i, 1:] = Y_train[i, :-1]
        # decoder_target_data_train[i] = Y_train[i]
        decoder_target_data_train[i, :-1] = Y_train[i, 1:]
    for i in range(len(Y_test)):
        decoder_input_data_test[i, 1:] = Y_test[i, :-1]
        # decoder_target_data_test[i] = Y_test[i]
        decoder_target_data_test[i, :-1] = Y_test[i, 1:]

    return decoder_input_data_train, decoder_target_data_train, decoder_input_data_test, decoder_target_data_test

def __preprocess_keypoints(train_parser: tvb_hksl_split_parser, test_parser: tvb_hksl_split_parser, keypoint_dir: str):
    X_train, Y_train, X_test, Y_test = __keypoints_reading(train_parser, test_parser, keypoint_dir)
    word_dict = generate_word_dict([Y_train, Y_test])
    Y_train, Y_test = __map_words_to_integers(Y_train, Y_test, word_dict)
    X_train, Y_train, X_test, Y_test = __convert_to_numpy_array(X_train, Y_train, X_test, Y_test)
    decoder_input_data_train, decoder_target_data_train, decoder_input_data_test, decoder_target_data_test = __create_decoder_data(Y_train, Y_test)
    return X_train, Y_train, X_test, Y_test, word_dict, decoder_input_data_train, decoder_target_data_train, decoder_input_data_test, decoder_target_data_test

def get_provided_keypoints(train_parser: tvb_hksl_split_parser, test_parser: tvb_hksl_split_parser, keypoint_dir: str):
    X_train, Y_train, X_test, Y_test, word_dict = __preprocess_keypoints(train_parser, test_parser, keypoint_dir)

    # Remove the 3rd dimension of X_train and X_test
    X_train = X_train[:, :, :, :2]
    X_test = X_test[:, :, :, :2]

    print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape, sep="\n")
    """
    dev:
    (322, 376, 133, 2)
    (322, 43)
    (322, 376, 133, 2)
    (322, 43)
    train:
    (6516, 376, 133, 2)
    (6516, 45)
    (322, 376, 133, 2)
    (322, 45)

    X: (
        number of samples, 
        number of frames, (highest number of frames among all samples in the dataset)
        number of keypoints, (68 face, 42 hands, 11 upper body)
        coordinates (x, y, 0)
    )
    Y: (
        number of samples,
        number of words in the glosses
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

    # shift the values to be positive
    min_value = min(np.min(X_train), np.min(X_test))
    shift_value = abs(min_value)
    X_train += shift_value
    X_test += shift_value

    # normalize the values
    # X_train = X_train / np.max(X_train)
    # X_test = X_test / np.max(X_test)
    # exit()

    # reshape X from 4D to 3D
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], -1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], -1)

    return X_train, Y_train, X_test, Y_test, word_dict

def get_mediapipe_keypoints_index():
    """
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

    For all keypoints, we only need to keep the x, y, z values
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

def get_mediapipe_keypoints(train_parser: tvb_hksl_split_parser, test_parser: tvb_hksl_split_parser, keypoint_dir: str):
    KEYPOINTS_INDEX = get_mediapipe_keypoints_index()

    # print(KEYPOINTS_INDEX)
    # print(len(KEYPOINTS_INDEX)) # 312
    # exit()

    X_train, Y_train, X_test, Y_test, word_dict = __preprocess_keypoints(train_parser, test_parser, keypoint_dir)

    X_train = X_train[:, :, KEYPOINTS_INDEX]
    X_test = X_test[:, :, KEYPOINTS_INDEX]

    print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape, sep="\n")
    print("Total number of words:", len(word_dict))

    # Since all keypoint values are in the range of 0 to 1, we don't need to normalize the values
    # TODO: standardize the values if necessary (or if we include sources outside of the dataset)
    # No need to reshape 4D to 3D
    return X_train, Y_train, X_test, Y_test, word_dict