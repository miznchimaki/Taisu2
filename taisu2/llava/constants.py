CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "."

# Model Constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

# Constants for Qwen2 Tokenizer (IternVL-2.5 & InternVL-3)
IMG_START_TOKEN = "<img>"
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
IMG_END_TOKEN = "</img>"

# Constants for image processor & dynamic resolution strategy
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Constants for webdataset
SHARD_SHUFFLE_BUFSIZE = 2000
SHARD_SHUFFLE_INITIAL = 500
SAMPLE_SHUFFLE_BUFSIZE = 50000
SAMPLE_SHUFFLE_INITIAL = 1000

# Constants for Taisu2 chat
VQA_QUESTION_START_TOKEN = "<|vqa_question_start|>"
VQA_QUESTION_END_TOKEN = "<|vqa_question_end|>"
VQA_ANSWER_START_TOKEN = "<|vqa_answer_start|>"
VQA_ANSWER_END_TOKEN = "<|vqa_answer_end|>"
MULTITURN_SEP_TOKEN = "<|multiturn_sep|>"
NATIVE_CAPTION_START_TOKEN = "<|native_caption_start|>"
NATIVE_CAPTION_END_TOKEN = "<|native_caption_end|>"
RECAPTION_START_TOKEN = "<|recaption_start|>"
RECAPTION_END_TOKEN = "<|recaption_end|>"
