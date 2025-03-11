MODE = "dev" # train | dev

BATCH_SIZE = 32

CACHE_DIR = f"/mnt/d/dickmwong3/cache/{MODE}_2"

# set both numbers to 0 if you want to start from the beginning
checkpoint = {
    "iteration": 9,
    "counter": 100
}

import os
import pandas as pd
import numpy as np
import cv2
from mediapipe.python.solutions.holistic import Holistic
import keras
from concurrent.futures import ThreadPoolExecutor
from functools import partial

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] ="3"
os.makedirs(CACHE_DIR, exist_ok=True)

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
        self.train_info["glosses_tokenized"] = self.train_info["glosses_tokenized"].apply(lambda x: ["<START>"] + x + ["<END>"])
        

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

    # removed bc it returns a duplicate, not by memory reference
    # def get_full_info(self) -> pd.DataFrame:
    #     return self.train_info
    
    def get_word_dict(self) -> dict:
        word_dict = {}
        for tokens in self.train_info["glosses_tokenized"]:
            for token in tokens:
                if token not in word_dict:
                    word_dict[token] = len(word_dict)
        return word_dict

    def rare_token_reduction(self, token_freq) -> None:
        # create a dictionary of all tokens and their frequencies
        # token_freq = {}
        # for tokens in self.train_info["glosses_tokenized"]:
        #     for token in tokens:
        #         if token in token_freq:
        #             token_freq[token] += 1
        #         else:
        #             token_freq[token] = 1

        # simpler approach: if any token has a frequence of < 5, replace that token with <UNK>
        def replace_rare_tokens(tokens):
            return ["<UNK>" if token_freq[token] < 5 else token for token in tokens]
        self.train_info["glosses_tokenized"] = self.train_info["glosses_tokenized"].apply(replace_rare_tokens)

    def rare_sample_reduction(self, token_freq) -> None:
        # remove samples with words that satisfy token_freq[token] = 1
        self.train_info = self.train_info[self.train_info["glosses_tokenized"].apply(lambda x: any([token_freq[token] < 5 if token in token_freq else True for token in x]))]

train_parser = tvb_hksl_split_parser("../dataset/tvb-hksl-news/split/train.csv")
test_parser = tvb_hksl_split_parser("../dataset/tvb-hksl-news/split/test.csv")
dev_parser = tvb_hksl_split_parser("../dataset/tvb-hksl-news/split/dev.csv")

# sample preprocessing
# train_parser.rare_sample_reduction(token_freq)
# test_parser.rare_sample_reduction(token_freq)
# dev_parser.rare_sample_reduction(token_freq)

if MODE == "train":
    actual_train_parser = train_parser
elif MODE == "dev":
    actual_train_parser = dev_parser

# if a word in test_parser is not in dev_parser or train_parser, remove that sample
# this is to prevent the model from predicting words that are not in the training set
# supposed, this should not happen, but just in case
parser_word_dict = actual_train_parser.get_word_dict()
test_parser.train_info = test_parser.train_info[test_parser.train_info["glosses_tokenized"].apply(lambda x: all([word in parser_word_dict for word in x]))]
    
# assert that all words in test_parser are also in train_parser
test_word_dict = test_parser.get_word_dict()
assert all([word in parser_word_dict for word in test_word_dict])

# finally, print the number of samples in each parser
print(f"train_parser: {len(train_parser.train_info)}")
print(f"test_parser: {len(test_parser.train_info)}")
print(f"dev_parser: {len(dev_parser.train_info)}")

# based on the dictionary, create one-hot vectors for each word
# then create decoder inputs and targets
all_input_found = False
# attempt to load the cached data if it exists
path_requirements = [
    f"../cache/train_decoder_input.npy",
    f"../cache/train_decoder_target.npy",
    f"../cache/test_decoder_input.npy",
    f"../cache/test_decoder_target.npy",
    f"../cache/dev_decoder_input.npy",
    f"../cache/dev_decoder_target.npy"
]

if all([os.path.exists(path) for path in path_requirements]):
    train_decoder_input = np.load(f"../cache/train_decoder_input.npy", mmap_mode="r")
    train_decoder_target = np.load(f"../cache/train_decoder_target.npy", mmap_mode="r")
    test_decoder_input = np.load(f"../cache/test_decoder_input.npy", mmap_mode="r")
    test_decoder_target = np.load(f"../cache/test_decoder_target.npy", mmap_mode="r")
    dev_decoder_input = np.load(f"../cache/dev_decoder_input.npy", mmap_mode="r")
    dev_decoder_target = np.load(f"../cache/dev_decoder_target.npy", mmap_mode="r")
    all_input_found = True
    print("All cached data found.")
else: 
    print("Cached data not found.")
    raise FileNotFoundError("Cached data not found.")

def pad_to_size(image, target_size=(1920, 1080)):
    rows, cols, _ = image.shape
    target_cols, target_rows = target_size
    scale_factor = min(target_cols / cols, target_rows / rows)
    new_cols = int(cols * scale_factor)
    new_rows = int(rows * scale_factor)
    resized_image = cv2.resize(image, (new_cols, new_rows), interpolation=cv2.INTER_LINEAR)

    top = (target_rows - new_rows) // 2
    bottom = target_rows - new_rows - top
    left = (target_cols - new_cols) // 2
    right = target_cols - new_cols - left
    padded_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    return padded_image

def apply_random_transformation(image, angle, tx, ty, scale):
    rows, cols, _ = image.shape

    # rotation
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    rotated_image = cv2.warpAffine(image, M, (cols, rows))

    # translation
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    translated_image = cv2.warpAffine(rotated_image, M, (cols, rows))

    # scaling
    scaled_image = cv2.resize(translated_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    # consistency
    final_image = cv2.resize(scaled_image, (cols, rows), interpolation=cv2.INTER_LINEAR)

    return final_image

# def ensure_in_bounds(image, target_size=(1920, 1080)):
#     rows, cols, _ = image.shape
#     target_cols, target_rows = target_size

#     # create a black canvas of the target size
#     canvas = np.zeros((target_rows, target_cols, 3), dtype=np.uint8)

#     # find the bounding box of non-black pixels
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
#     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     x, y, w, h = cv2.boundingRect(contours[0])

#     top = (target_rows - h) // 2
#     left = (target_cols - w) // 2
#     canvas[top:top+h, left:left+w] = image[y:y+h, x:x+w]

#     return canvas

def preprocess_image(image, target_size=(1920, 1080), angle=0, tx=0, ty=0, scale=1):
    padded_image = pad_to_size(image, target_size)
    transformed_image = apply_random_transformation(padded_image, angle, tx, ty, scale)
    # final_image = ensure_in_bounds(transformed_image, target_size)
    # return final_image
    return transformed_image


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
        """
        # pose with visibility
        # POSE = list(range(0, 33*4))

        # pose without visibility
        POSE_UNPROCESSED = range(0, 33*4)
        # POSE = [i for i in POSE_UNPROCESSED if i % 4 != 3]
        # for x, y only
        # discard Z due to documentation https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/holistic.md
        POSE = [i for i in POSE_UNPROCESSED if i % 4 != 2 and i % 4 != 3]

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

STATIC_KEYPOINTS_INDEX = get_mediapipe_keypoints_index() # saves time by not recalculating the indices

def mediapipe_detection(frame, holistic):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = holistic.process(frame)
    # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return result

def mediapipe_extract_keypoints(result):
    # print(len(result.pose_landmarks.landmark))
    # print(len(result.face_landmarks.landmark))
    # print(len(result.left_hand_landmarks.landmark))
    # print(len(result.right_hand_landmarks.landmark))
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in result.pose_landmarks.landmark]).flatten() if result.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in result.face_landmarks.landmark]).flatten() if result.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in result.left_hand_landmarks.landmark]).flatten() if result.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in result.right_hand_landmarks.landmark]).flatten() if result.right_hand_landmarks else np.zeros(21*3)
    concat = np.concatenate([pose, face, lh, rh])
    return concat[STATIC_KEYPOINTS_INDEX]

def mediapipe_extract(frames, override_random = False): # frames is a list of frames
    if override_random:
        angle = 0
        tx = 0
        ty = 0
        scale = 1
    else:
        # predefine random transformation parameters
        angle = np.random.uniform(-10, 10)
        tx = np.random.uniform(-100, 100)
        ty = np.random.uniform(-60, 60)
        scale = np.random.uniform(0.6, 1.2)
    frames = [preprocess_image(frame, angle=angle, tx=tx, ty=ty, scale=scale) for frame in frames]
    # one optimization done here is to use the same holistic object for all frames
    # this way, the model only needs to be loaded once
    # then keypoints can be tracked until not found
    with Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=0) as holistic:
        return [mediapipe_extract_keypoints(mediapipe_detection(frame, holistic)) for frame in frames]
    # with Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    #     with ThreadPoolExecutor(max_workers=2) as executor:
    #         results = list(executor.map(mediapipe_detection, frames, [holistic] * len(frames)))
    #     return [mediapipe_extract_keypoints(result) for result in results]

# Preparation: get the largest length of sequences of x
train_max_length = train_parser.get_max_length()
test_max_length = test_parser.get_max_length()
dev_max_length = dev_parser.get_max_length()

X_max_length = max(train_max_length, test_max_length, dev_max_length)
print("Max length of sequences of X:", X_max_length)

class KeypointGenerator(keras.utils.Sequence):
    def __init__(self, X, decoder_input, decoder_target, X_max_length, batch_size=32):
        self.x = X
        self.decoder_input = decoder_input
        self.decoder_target = decoder_target
        self.X_max_length = X_max_length
        self.batch_size = batch_size

    def __len__(self):
        # return len(self.x) // self.batch_size
        return (len(self.x) + self.batch_size - 1) // self.batch_size
    
    def __preprocess_x__(self, dir_location):
        # x is the train id. the directory of the frames is located at ../dataset/tvb-hksl-news/frames/{train_id}
        source_directory = f"../dataset/tvb-hksl-news/frames/{dir_location}"
        # print(sorted(os.listdir(source_directory)))
        source_list_frames = [cv2.imread(os.path.join(source_directory, frame)) for frame in sorted(os.listdir(source_directory))]
        return source_list_frames

    def __getitem__(self, idx, override_random = False):
        start_idx = idx * self.batch_size
        end_idx = min((idx + 1) * self.batch_size, len(self.x))

        # edge case: start_idx = len(self.x), which means the previous batch was the last batch
        # if start_idx == len(self.x):
        #     return [None, None], None
        if start_idx >= len(self.x):
            raise IndexError("Index out of range for the generator")

        batch_x = self.x[start_idx:end_idx]
        batch_decoder_input = self.decoder_input[start_idx:end_idx]
        batch_decoder_target = self.decoder_target[start_idx:end_idx]
        # batch_x = [self.__preprocess_x__(dir_location) for dir_location in batch_x]
        # batch_x = [mediapipe_extract(frames) for frames in batch_x]
        with ThreadPoolExecutor(max_workers=4) as executor:
            batch_x = list(executor.map(self.__preprocess_x__, batch_x))
        partial_mediapipe_extract = partial(mediapipe_extract, override_random=override_random)
        with ThreadPoolExecutor(max_workers=4) as executor:
            # TODO: research for a faster method than mediapipe
            # batch_x = list(executor.map(mediapipe_extract, batch_x, override_random=[override_random] * len(batch_x)))
            batch_x = list(executor.map(partial_mediapipe_extract, batch_x))

        # # pad each sequence to the max length
        batch_x = keras.preprocessing.sequence.pad_sequences(batch_x, maxlen=self.X_max_length, padding="post", dtype="float32")
        batch_x = np.array(batch_x)
        batch_decoder_input = np.array(batch_decoder_input)
        batch_decoder_target = np.array(batch_decoder_target)
        return (batch_x, batch_decoder_input), batch_decoder_target
    
    # iteration value is only used for pre-generating and caching the data
if MODE == "train":
    keypoint_generator = KeypointGenerator(train_parser.get_train_id(), train_decoder_input, train_decoder_target, X_max_length, batch_size=BATCH_SIZE)
    iteration = 19
elif MODE == "dev":
    keypoint_generator = KeypointGenerator(dev_parser.get_train_id(), dev_decoder_input, dev_decoder_target, X_max_length, batch_size=BATCH_SIZE)
    iteration = 0
else: raise ValueError("Invalid mode")

x_dir = os.path.join(CACHE_DIR, "x")
decoder_input_dir = os.path.join(CACHE_DIR, "decoder_input")
decoder_target_dir = os.path.join(CACHE_DIR, "decoder_target")

os.makedirs(x_dir, exist_ok=True)
os.makedirs(decoder_input_dir, exist_ok=True)
os.makedirs(decoder_target_dir, exist_ok=True)

# instant test
# start = time.time()
# (debug_x, debug_decoder_input), debug_decoder_target = keypoint_generator.__getitem__(0)
# print(debug_x.shape)
# print(debug_decoder_input.shape)
# print(debug_decoder_target.shape)
# end = time.time()
# print(f"Time taken: {end - start}")
# exit()



for i in range(iteration):
    if i < checkpoint["iteration"]:
        continue
    elif i == checkpoint["iteration"]:
        counter = checkpoint["counter"]
    else:
        counter = 0
        
    while True:
        print(f"Iteration {i}, Counter {counter}")
        # break if the generator is exhausted (i.e. output < batch_size)
        (batch_x, batch_decoder_input), batch_decoder_target = keypoint_generator.__getitem__(counter)
        if batch_x is None: # edge case only
            print("Generator exhausted")
            break
        file_name = f"iteration_{i}_batch_{counter}.npy"
        np.save(os.path.join(x_dir, file_name), batch_x)
        np.save(os.path.join(decoder_input_dir, file_name), batch_decoder_input)
        np.save(os.path.join(decoder_target_dir, file_name), batch_decoder_target)
        counter += 1
        batch_x_len = len(batch_x)
        if len(batch_x) < keypoint_generator.batch_size:
            print("Generator exhausted")
            break

# control generation: no transformation
counter = 0
while True:
    print(f"Control, Counter {counter}")
    (batch_x, batch_decoder_input), batch_decoder_target = keypoint_generator.__getitem__(counter, override_random=True)
    if batch_x is None:
        print("Generator exhausted")
        break
    file_name = f"control_batch_{counter}.npy"
    np.save(os.path.join(x_dir, file_name), batch_x)
    np.save(os.path.join(decoder_input_dir, file_name), batch_decoder_input)
    np.save(os.path.join(decoder_target_dir, file_name), batch_decoder_target)
    counter += 1
    batch_x_len = len(batch_x)
    if len(batch_x) < keypoint_generator.batch_size:
        print("Generator exhausted")
        break

# start time: 28/10/2024 01:33
# end time: 