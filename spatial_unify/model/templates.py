import torch

DEFAULT_IMAGE_PATCH_TOKEN = "<imagePatch>"
DEFAULT_IM_START_TOKEN = "<image>"
DEFAULT_IM_END_TOKEN = "</image>"
DEFAULT_VID_START_TOKEN = "<video>"
DEFAULT_VID_END_TOKEN = "</video>"
DEFAULT_GEN_IMAGE_PATCH_TOKEN = "<gen_imagePatch>"
DEFAULT_GEN_IM_START_TOKEN = "<gen_image>"
DEFAULT_GEN_IM_END_TOKEN = "</gen_image>"
PLACEHOLDER_IMAGE_TOKEN_IN_TEXT = "<imageHere>"
DEFAULT_END_OF_CHUNK_TOKEN = "<end_of_chunk>"

DEFAULT_END_OF_AUDIO_TOKEN = "<end_of_audio>"
DEFAULT_AUDIO_PATCH_TOKEN = "<audioPatch>"
DEFAULT_AU_START_TOKEN = "<audio>"
DEFAULT_AU_END_TOKEN = "</audio>"
DEFAULT_GEN_AUDIO_PATCH_TOKEN = "<gen_audioPatch>"
DEFAULT_GEN_AU_START_TOKEN = "<gen_audio>"
DEFAULT_GEN_AU_END_TOKEN = "</gen_audio>"
PLACEHOLDER_AUDIO_TOKEN_IN_TEXT = "<audioHere>"
DEFAULT_FRAME_PATCH_TOKEN = "<framePatch>"
DEFAULT_TEXT_TOKEN = '<text>'
DEFAULT_ASR_TOKEN = '<asr>'
DEFAULT_TTS_TOKEN = '<tts>'

BAILING_MM_TOKENS = [
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_VID_START_TOKEN,
    DEFAULT_VID_END_TOKEN,
    DEFAULT_GEN_IMAGE_PATCH_TOKEN,
    DEFAULT_GEN_IM_START_TOKEN,
    DEFAULT_GEN_IM_END_TOKEN,
    PLACEHOLDER_IMAGE_TOKEN_IN_TEXT,
    DEFAULT_END_OF_CHUNK_TOKEN,
    DEFAULT_END_OF_AUDIO_TOKEN,
    DEFAULT_AUDIO_PATCH_TOKEN,
    DEFAULT_AU_START_TOKEN,
    DEFAULT_AU_END_TOKEN,
    DEFAULT_GEN_AUDIO_PATCH_TOKEN,
    DEFAULT_GEN_AU_START_TOKEN,
    DEFAULT_GEN_AU_END_TOKEN,
    PLACEHOLDER_AUDIO_TOKEN_IN_TEXT,
    DEFAULT_FRAME_PATCH_TOKEN,
    DEFAULT_TEXT_TOKEN,
    DEFAULT_ASR_TOKEN,
    DEFAULT_TTS_TOKEN,
]

GLM_USER_PREFIX = "<role>HUMAN</role>"
GLM_ASSISTANT_PREFIX = "<role>ASSISTANT</role>"

SYSTEM_PROMPT = "You are a helpful language, vision and audio assistant. You are able to understand the visual and audio content that the user provides, and assist the user with a variety of tasks using natural language. "
SYSTEM_PROMPT_CN = "你是一个有帮助的语言,视觉和音频助手。你能理解用户提供的视觉与音频内容，并用自然语言帮助用户完成各种任务。 "


class Bailingv4Template:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.system_prefix_id = []
        self.usr_prefix = GLM_USER_PREFIX
        self.assistant_prefix = GLM_ASSISTANT_PREFIX

        self.usr_prefix_id = (
            self.tokenizer(self.usr_prefix, return_tensors="pt")["input_ids"][0]
        ).tolist()
        self.assistant_prefix_id = (
            self.tokenizer(self.assistant_prefix, return_tensors="pt")["input_ids"][0]
        ).tolist()
        bos_token = (
            self.tokenizer.bos_token
            if self.tokenizer.eos_token is None
            else self.tokenizer.pad_token
        )
        self.bos_id = self.tokenizer.convert_tokens_to_ids(bos_token)
        self.eos_id = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.eos_token
        )
        self.pad_id = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.pad_token
        )
        self.llm_eos_token = self.tokenizer.eos_token

    def handle_system_prompt(self, prompt):
        return "", [], []

    def handle_user_message(self, question, task_prompt_, task_prompt_id_):
        input_text = ""
        input_ids = []
        labels = []
        input_text += self.usr_prefix
        input_ids.extend(self.usr_prefix_id)
        labels.extend([-100] * len(self.usr_prefix_id))
        input_text += question + task_prompt_
        question_id = (
            self.tokenizer(question, return_tensors="pt")["input_ids"][0]
        ).tolist()
        input_ids.extend(question_id)
        input_ids.extend(task_prompt_id_)
        labels.extend([-100] * (len(question_id) + len(task_prompt_id_)))
        return input_text, input_ids, labels

    def handle_assistant_message(self, answer, is_last):
        input_text = ""
        input_ids = []
        labels = []
        input_text += self.assistant_prefix
        input_ids.extend(self.assistant_prefix_id)
        labels.extend([-100] * len(self.assistant_prefix_id))

        assistant_prefix_length = len(input_ids)

        input_text += answer
        answer_id = (
            self.tokenizer(answer, return_tensors="pt")["input_ids"][
                0
            ]
        ).tolist()

        input_ids.extend(answer_id)
        labels.extend(answer_id)

        if not is_last:
            input_text += self.tokenizer.eos_token
            input_ids.extend([self.eos_id])
            labels.extend([-100] * len([self.eos_id]))
            return input_text, input_ids, labels, 0, 0
        else:
            output_length = len(answer_id) + len([self.eos_id])
            return input_text, input_ids, labels, output_length, assistant_prefix_length

    def finalize_sequence(self, input_text, input_ids, labels, max_length, position_ids):
        num_special_tokens = 2

        if (
            len(input_ids) > max_length - num_special_tokens
        ):
            input_ids = input_ids[: max_length - num_special_tokens]
            labels = labels[: max_length - num_special_tokens]
        assert len(input_ids) <= max_length - num_special_tokens

        input_ids = torch.cat([
            torch.tensor(input_ids), torch.tensor([self.eos_id])]
        )
        labels = torch.cat([torch.tensor(labels[1:]), torch.tensor([self.eos_id, -100])]).to(
            torch.int64
        )
        position_ids = [position_id[: input_ids.shape[0]] for position_id in position_ids]
        attention_mask = None
        attention_mask_for_mrope = torch.ones_like(input_ids, dtype=torch.int64)

        input_text += self.tokenizer.eos_token

        return input_text, input_ids, labels, attention_mask, position_ids, attention_mask_for_mrope
