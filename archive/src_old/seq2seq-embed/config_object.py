import yaml
import os

class ConfigObject():
    """
    A class to handle configuration settings for the project.

    Reads a YAML configuration file and provide methods to access various configuration parameters.
    It also ensures that certain paths exist and creates necessary directories if not.

    Attributes:
        __config (dict): The configuration dictionary loaded from the YAML file.
        __kp_type (str): The type of keypoints, either "provided" or "mediapipe".
        __keypoint_dir (str): The directory where keypoints are stored.
        __model_path (str): The path to the model.
        __encoder_path (str): The path to the encoder model.
        __decoder_path (str): The path to the decoder model.
        __checkpoint_path (str): The path to model checkpoint.
        __train_file (str): The path to training split file.
        __test_file (str): The path to testing split file.
        __cache_dir (str): The path to cache directory. Note that the code will always store keypoints in cache directory.

    Methods:
        get_type() -> str: 
            Returns the type of keypoints.
        get_keypoint_dir() -> str: 
            Returns the directory where keypoints are stored.
        get_model_path() -> str: 
            Returns the path to the model.
        get_encoder_path() -> str: 
            Returns the path to the encoder model.
        get_decoder_path() -> str: 
            Returns the path to the decoder model.
        get_checkpoint_path() -> str: 
            Returns the path to the checkpoint model.
        get_train_file() -> str: 
            Returns the path to the split training file.
        get_test_file() -> str: 
            Returns the path to the split testing file.
        get_cache_dir() -> str: 
            Returns the path to the cache directory.
        get_word_dict_path() -> str: 
            Returns the path to the word dictionary file.
    """

    def __init__(self, config_path):
        with open(config_path, "r") as f:
            self.__config: dict[str, any] = yaml.safe_load(f)

        # assert all inputs are strings
        for key in self.__config:
            assert isinstance(self.__config[key], str)

        self.__kp_type: str = self.__config["type"]
        self.__keypoint_dir: str = self.__config["keypoint_dir"]
        self.__model_path: str = self.__config["model_path"]
        self.__encoder_path: str = self.__config["encoder_path"]
        self.__decoder_path: str = self.__config["decoder_path"]
        self.__checkpoint_path: str = self.__config["checkpoint_path"]
        self.__train_file: str = self.__config["train_file"]
        self.__test_file: str = self.__config["test_file"]
        self.__cache_dir: str = self.__config["cache_dir"]

        # assert type is either "provided" or "mediapipe"
        assert self.__kp_type == "provided" or self.__kp_type == "mediapipe"

        # assert all the non-generative paths are valid
        # print(self.__keypoint_dir)
        assert os.path.exists(self.__keypoint_dir)
        assert os.path.exists(self.__train_file)

        # generate the directory for the word_dict_path and model_path
        os.makedirs(os.path.dirname(self.__model_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.__encoder_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.__decoder_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.__checkpoint_path), exist_ok=True)

    def get_type(self) -> str:
        """
        Returns the type of keypoints.

        Returns:
            str: The type of keypoints, either "provided" or "mediapipe".
        """
        return self.__kp_type

    def get_keypoint_dir(self) -> str:
        """
        Returns the directory where keypoints are stored.

        Returns:
            str: The directory path for keypoints.
        """
        return self.__keypoint_dir

    def get_model_path(self) -> str:
        """
        Returns the path to the model file.

        Returns:
            str: The path to the model file.
        """
        return self.__model_path

    def get_encoder_path(self) -> str:
        """
        Returns the path to the encoder file.

        Returns:
            str: The path to the encoder file.
        """
        return self.__encoder_path

    def get_decoder_path(self) -> str:
        """
        Returns the path to the decoder file.

        Returns:
            str: The path to the decoder file.
        """
        return self.__decoder_path

    def get_checkpoint_path(self) -> str:
        """
        Returns the path to the checkpoint file.

        Returns:
            str: The path to the checkpoint file.
        """
        return self.__checkpoint_path

    def get_train_file(self) -> str:
        """
        Returns the path to the training file.

        Returns:
            str: The path to the training file.
        """
        return self.__train_file

    def get_test_file(self) -> str:
        """
        Returns the path to the testing file.

        Returns:
            str: The path to the testing file.
        """
        return self.__test_file

    def get_cache_dir(self) -> str:
        """
        Returns the path to the cache directory.

        Returns:
            str: The path to the cache directory.
        """
        return self.__cache_dir
    
    def get_word_dict_path(self) -> str:
        """
        Returns the path to the word dictionary file.

        Returns:
            str: The path to the word dictionary file.
        """
        return os.path.join(self.__cache_dir, "word_dict.json")