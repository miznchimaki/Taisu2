import dataclasses
from enum import auto, IntEnum
from typing import Dict, List, Tuple, Union


class SeparatorStyle(IntEnum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()
    PLAIN = auto()
    LLAMA_2 = auto()

    # separators for InternVL-2.5/InternVL-3
    ADD_COLON_SINGLE = auto()
    ADD_COLON_TWO = auto()
    ADD_COLON_SPACE_SINGLE = auto()
    NO_COLON_SINGLE = auto()
    NO_COLON_TWO = auto()
    ADD_NEW_LINE_SINGLE = auto()
    CHATGLM = auto()
    CHATML = auto()
    CHATINTERN = auto()
    DOLLY = auto()
    RWKV = auto()
    PHOENIX = auto()
    ROBIN = auto()
    FALCON_CHAT = auto()
    CHATGLM3 = auto()
    INTERNVL_ZH = auto()
    MPT = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    name: str
    system_template: str = '{system_message}'
    system_message: str = ''
    roles: Tuple[str] = ('user', 'gpt')
    messages: List[List[str]] = ()
    offset: int = 0
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None
    # Stop criteria (the default one is EOS token)
    stop_str: Union[str, List[str]] = None
    # Stops generation if meeting any token in this list
    stop_token_ids: List[int] = None

    def get_prompt(self) -> str:
        system_prompt = self.system_template.format(system_message=self.system_message)
        if len(self.messages) > 0 and type(self.messages[0][1]) is tuple:
            init_role, init_msg = self.messages[0].copy()
            init_msg = init_msg[0].replace("<image>", "").strip()
            self.messages[0] = (init_role, "<image>\n" + init_msg)

        if self.sep_style == SeparatorStyle.SINGLE:
            ret = system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = system_prompt + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
        elif self.sep_style == SeparatorStyle.LLAMA_2:
            wrap_sys = lambda msg: f"<<SYS>>\n{msg}\n<</SYS>>\n\n"
            wrap_inst = lambda msg: f"[INST] {msg} [/INST]"
            ret = ""

            for i, (role, message) in enumerate(self.messages):
                if i == 0:
                    assert message, "first message should not be none"
                    assert role == self.roles[0], "first message should come from user"
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    if i == 0: message = wrap_sys(system_prompt) + message
                    if i % 2 == 0:
                        message = wrap_inst(message)
                        ret += self.sep + message
                    else:
                        ret += " " + message + " " + self.sep2
                else:
                    ret += ""
            ret = ret.lstrip(self.sep)
        elif self.sep_style == SeparatorStyle.PLAIN:
            seps = [self.sep, self.sep2]
            ret = system_prompt
            for i, (role, message) in enumerate(self.messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += message + seps[i % 2]
                else:
                    ret += ""
        # Below conversation separators are all for InternVL2.5/InternVL-3
        elif self.sep_style == SeparatorStyle.ADD_COLON_SINGLE:
            ret = system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ': ' + message + self.sep
                else:
                    ret += role + ':'
        elif self.sep_style == SeparatorStyle.ADD_COLON_TWO:
            seps = [self.sep, self.sep2]
            ret = system_prompt + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ': ' + message + seps[i % 2]
                else:
                    ret += role + ':'
        elif self.sep_style == SeparatorStyle.ADD_COLON_SPACE_SINGLE:
            ret = system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ': ' + message + self.sep
                else:
                    ret += role + ': '  # must be end with a space
        elif self.sep_style == SeparatorStyle.ADD_NEW_LINE_SINGLE:
            ret = '' if system_prompt == '' else system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + '\n' + message + self.sep
                else:
                    ret += role + '\n'
        elif self.sep_style == SeparatorStyle.NO_COLON_SINGLE:
            ret = system_prompt
            for role, message in self.messages:
                if message:
                    ret += role + message + self.sep
                else:
                    ret += role
        elif self.sep_style == SeparatorStyle.NO_COLON_TWO:
            seps = [self.sep, self.sep2]
            ret = system_prompt
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + message + seps[i % 2]
                else:
                    ret += role
        elif self.sep_style == SeparatorStyle.RWKV:
            ret = system_prompt
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += (
                        role
                        + ': '
                        + message.replace('\r\n', '\n').replace('\n\n', '\n')
                    )
                    ret += '\n\n'
                else:
                    ret += role + ':'
        elif self.sep_style == SeparatorStyle.CHATGLM:
            # source: https://huggingface.co/THUDM/chatglm-6b/blob/1d240ba371910e9282298d4592532d7f0f3e9f3e/modeling_chatglm.py#L1302-L1308
            # source2: https://huggingface.co/THUDM/chatglm2-6b/blob/e186c891cf64310ac66ef10a87e6635fa6c2a579/modeling_chatglm.py#L926
            round_add_n = 1 if self.name == 'chatglm2' else 0
            if system_prompt:
                ret = system_prompt + self.sep
            else:
                ret = ''

            for i, (role, message) in enumerate(self.messages):
                if i % 2 == 0:
                    ret += f'[Round {i//2 + round_add_n}]{self.sep}'

                if message:
                    ret += f'{role}：{message}{self.sep}'
                else:
                    ret += f'{role}：'
        elif self.sep_style == SeparatorStyle.CHATGLM3:
            ret = ''
            if self.system_message:
                ret += system_prompt
            for role, message in self.messages:
                if message:
                    ret += role + '\n' + ' ' + message
                else:
                    ret += role
        elif self.sep_style == SeparatorStyle.CHATINTERN:
            # source: https://huggingface.co/internlm/internlm-chat-7b-8k/blob/bd546fa984b4b0b86958f56bf37f94aa75ab8831/modeling_internlm.py#L771
            seps = [self.sep, self.sep2]
            ret = system_prompt
            for i, (role, message) in enumerate(self.messages):
                # if i % 2 == 0:
                #     ret += "<s>"
                if message:
                    ret += role + ':' + message + seps[i % 2] + '\n'
                else:
                    ret += role + ':'
        elif self.sep_style == SeparatorStyle.DOLLY:
            seps = [self.sep, self.sep2]
            ret = system_prompt
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ':\n' + message + seps[i % 2]
                    if i % 2 == 1:
                        ret += '\n\n'
                else:
                    ret += role + ':\n'
        elif self.sep_style == SeparatorStyle.PHOENIX:
            ret = system_prompt
            for role, message in self.messages:
                if message:
                    ret += role + ': ' + '<s>' + message + '</s>'
                else:
                    ret += role + ': ' + '<s>'
        elif self.sep_style == SeparatorStyle.ROBIN:
            ret = system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ':\n' + message + self.sep
                else:
                    ret += role + ':\n'
        elif self.sep_style == SeparatorStyle.FALCON_CHAT:
            ret = ''
            if self.system_message:
                ret += system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ': ' + message + self.sep
                else:
                    ret += role + ':'
        elif self.sep_style == SeparatorStyle.INTERNVL_ZH:
            seps = [self.sep, self.sep2]
            ret = self.system_message + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ': ' + message + seps[i % 2]
                else:
                    ret += role + ':'
        elif self.sep_style == SeparatorStyle.MPT:
            ret = system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + message + self.sep
                else:
                    ret += role
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

        return ret

    def set_system_message(self, system_message: str):
        """Set the system message."""
        self.system_message = system_message

    def append_message(self, role: str, message: str):
        """Append a new message."""
        self.messages.append([role, message])

    def update_last_message(self, message: str):
        """Update the last output.

        The last message is typically set to be None when constructing the prompt,
        so we need to update it in-place after getting the response from a model.
        """
        self.messages[-1][1] = message

    def get_images(self, return_pil=False, return_org=False):
        images = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    import base64
                    from io import BytesIO
                    from PIL import Image
                    msg, image, image_process_mode = msg
                    org_image = image.copy()
                    print(f"image_process_mode: {image_process_mode}")
                    if image_process_mode == "Pad":
                        def expand2square(pil_img, background_color=(122, 116, 104)):
                            width, height = pil_img.size
                            if width == height:
                                return pil_img
                            elif width > height:
                                result = Image.new(pil_img.mode, (width, width), background_color)
                                result.paste(pil_img, (0, (width - height) // 2))
                                return result
                            else:
                                result = Image.new(pil_img.mode, (height, height), background_color)
                                result.paste(pil_img, ((height - width) // 2, 0))
                                return result
                        image = expand2square(image)
                    elif image_process_mode in ["Default", "Crop"]:
                        pass
                    elif image_process_mode == "Resize":
                        image = image.resize((336, 336))
                    else:
                        raise ValueError(f"Invalid image_process_mode: {image_process_mode}")

                    resize_image_flag = False
                    if resize_image_flag:
                        max_hw, min_hw = max(image.size), min(image.size)
                        aspect_ratio = max_hw / min_hw
                        max_len, min_len = 896, 448
                        shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))
                        longest_edge = int(shortest_edge * aspect_ratio)
                        W, H = image.size
                        if longest_edge != max(image.size):
                            if H > W:
                                H, W = longest_edge, shortest_edge
                            else:
                                H, W = shortest_edge, longest_edge
                            image = image.resize((W, H))
                    if return_pil:
                        if return_org:
                            images.append(image)
                        else:
                            images.append(org_image)
                    else:
                        buffered = BytesIO()
                        if return_org:
                            org_image.save(buffered, format="JPEG")
                        else:
                            image.save(buffered, format="PNG")
                        img_b64_str = base64.b64encode(buffered.getvalue()).decode()
                        images.append(img_b64_str)
        return images

    def to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    import base64
                    from io import BytesIO
                    msg, image, image_process_mode = msg
                    max_hw, min_hw = max(image.size), min(image.size)
                    aspect_ratio = max_hw / min_hw
                    max_len, min_len = 800, 400
                    shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))
                    longest_edge = int(shortest_edge * aspect_ratio)
                    W, H = image.size
                    if H > W:
                        H, W = longest_edge, shortest_edge
                    else:
                        H, W = shortest_edge, longest_edge
                    image = image.resize((W, H))
                    buffered = BytesIO()
                    image.save(buffered, format="JPEG")
                    img_b64_str = base64.b64encode(buffered.getvalue()).decode()
                    img_str = f'<img src="data:image/png;base64,{img_b64_str}" alt="user upload image" />'
                    msg = img_str + msg.replace('<image>', '').strip()
                    ret.append([msg, None])
                else:
                    ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def copy(self):
        return Conversation(
            name=self.name, 
            system_template=self.system_template, 
            system_message=self.system_message, 
            roles=self.roles, 
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            stop_str=self.stop_str,
            stop_token_ids=self.stop_token_ids
            )

    def dict(self):
        return {
            "template_name": self.name, 
            "system_message": self.system_message, 
            "roles": self.roles, 
            "message": self.messages, 
            "offset": self.offset, 
        }


conv_templates: Dict[str, Conversation] = {}


def register_conv_template(template: Conversation, override: bool = False):
    """Register a new conversation template."""
    if not override:
        assert (
            template.name not in conv_templates
        ), f'{template.name} has been registered.'

    conv_templates[template.name] = template


def get_conv_template(name: str) -> Conversation:
    """Get a conversation template."""
    return conv_templates[name].copy()


conv_vicuna_v0 = Conversation(
    name="v0",
    system_message=(
                    "A chat between a curious human and an artificial intelligence assistant. "
                    "The assistant gives helpful, detailed, and polite answers to the human's questions."
                   ), 
    roles=("Human", "Assistant"),
    messages=[
        ["Human", "What are the key differences between renewable and non-renewable energy sources?"],
        ["Assistant",
            "Renewable energy sources are those that can be replenished naturally in a relatively "
            "short amount of time, such as solar, wind, hydro, geothermal, and biomass. "
            "Non-renewable energy sources, on the other hand, are finite and will eventually be "
            "depleted, such as coal, oil, and natural gas. Here are some key differences between "
            "renewable and non-renewable energy sources:\n"
            "1. Availability: Renewable energy sources are virtually inexhaustible, while non-renewable "
            "energy sources are finite and will eventually run out.\n"
            "2. Environmental impact: Renewable energy sources have a much lower environmental impact "
            "than non-renewable sources, which can lead to air and water pollution, greenhouse gas emissions, "
            "and other negative effects.\n"
            "3. Cost: Renewable energy sources can be more expensive to initially set up, but they typically "
            "have lower operational costs than non-renewable sources.\n"
            "4. Reliability: Renewable energy sources are often more reliable and can be used in more remote "
            "locations than non-renewable sources.\n"
            "5. Flexibility: Renewable energy sources are often more flexible and can be adapted to different "
            "situations and needs, while non-renewable sources are more rigid and inflexible.\n"
            "6. Sustainability: Renewable energy sources are more sustainable over the long term, while "
            "non-renewable sources are not, and their depletion can lead to economic and social instability.\n"]
    ],
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)
register_conv_template(conv_vicuna_v0)


conv_vicuna_v1 = Conversation(
    name="v1", 
    system_message=(
                    "A chat between a curious user and an artificial intelligence assistant. "
                    "The assistant gives helpful, detailed, and polite answers to the user's questions."
                   ),
    roles=("USER", "ASSISTANT"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)
register_conv_template(conv_vicuna_v1)


conv_vicuna_v1 = Conversation(
    name="vicuna_v1", 
    system_message=(
                    "A chat between a curious user and an artificial intelligence assistant. "
                    "The assistant gives helpful, detailed, and polite answers to the user's questions."
                   ),
    roles=("USER", "ASSISTANT"),
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)
register_conv_template(conv_vicuna_v1)


# Internlm-chat template
conv_internlm = Conversation(
    name="internlm",
    system_message=(
                    "A chat between a curious <|User|> and an <|Bot|>. The <|Bot|> gives helpful, "
                    "detailed, and polite answers to the <|User|>'s questions.\n\n"
                   ),
    roles=("<|User|>", "<|Bot|>"),
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.CHATINTERN,
    sep="<eoh>",
    sep2="<eoa>",
    stop_token_ids=[1, 103028],
    stop_str="<|User|>",
)
register_conv_template(conv_internlm)


conv_llama_2 = Conversation(
    name="llama_2", 
    system_message=(
                    "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  "
                    "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. "
                    "Please ensure that your responses are socially unbiased and positive in nature.\n\n"
                    "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. "
                    "If you don't know the answer to a question, please don't share false information."
                   ),
    roles=("USER", "ASSISTANT"),
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.LLAMA_2,
    sep="<s>",
    sep2="</s>",
)
register_conv_template(conv_llama_2)


conv_llava_llama_2 = Conversation(
    name="llava_llama_2", 
    system_message=(
                    "You are a helpful language and vision assistant. "
                    "You are able to understand the visual content that the user provides, "
                    "and assist the user with a variety of tasks using natural language.",
                   ),
    roles=("USER", "ASSISTANT"),
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.LLAMA_2,
    sep="<s>",
    sep2="</s>",
)
register_conv_template(conv_llava_llama_2)


conv_mpt = Conversation(
    name="mpt", 
    system_message=(
                    "<|im_start|>system\n"
                    "A conversation between a user and an LLM-based AI assistant. "
                    "The assistant gives helpful and honest answers."
                   ),
    roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.MPT,
    sep="<|im_end|>",
)
register_conv_template(conv_mpt)


conv_llava_plain = Conversation(
    name="plain", 
    system_message="",
    roles=("", ""),
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.PLAIN,
    sep="\n",
)
register_conv_template(conv_llava_plain)


conv_llava_plain = Conversation(
    name="v0_plain", 
    system_message="",
    roles=("", ""),
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.PLAIN,
    sep="\n",
)
register_conv_template(conv_llava_plain)


conv_llava_v0 = Conversation(
    name="llava_v0", 
    system_message=(
                    "A chat between a curious human and an artificial intelligence assistant. "
                    "The assistant gives helpful, detailed, and polite answers to the human's questions."
                   ),
    roles=("Human", "Assistant"),
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)
register_conv_template(conv_llava_v0)


conv_llava_v0_mmtag = Conversation(
    name="v0_mmtag", 
    system_message=(
                    "A chat between a curious user and an artificial intelligence assistant. "
                    "The assistant is able to understand the visual content that the user provides, "
                    "and assist the user with a variety of tasks using natural language."
                    "The visual content will be provided with the following format: <Image>visual content</Image>."
                   ),
    roles=("Human", "Assistant"),
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)
register_conv_template(conv_llava_v0_mmtag)


conv_llava_v1 = Conversation(
    name="llava_v1", 
    system_message=(
                    "A chat between a curious human and an artificial intelligence assistant. "
                    "The assistant gives helpful, detailed, and polite answers to the human's questions."
                   ),
    roles=("USER", "ASSISTANT"),
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)
register_conv_template(conv_llava_v1)


conv_llava_v1_mmtag = Conversation(
    name="v1_mmtag", 
    system_message=(
            "A chat between a curious user and an artificial intelligence assistant. "
            "The assistant is able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language."
            "The visual content will be provided with the following format: <Image>visual content</Image>."
           ),
    roles=("USER", "ASSISTANT"),
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)
register_conv_template(conv_llava_v1_mmtag)


internvl_zh = Conversation(
    name="internvl_zh", 
    system_message="",
    roles=("<human>", "<bot>"),
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.INTERNVL_ZH,
    sep=" ",
    sep2="</s>",
)
register_conv_template(internvl_zh)


hermes_2 = Conversation(
    name='Hermes-2',
    system_template='<|im_start|>system\n{system_message}',
    # note: The new system prompt was not used here to avoid changes in benchmark performance.
    # system_message='我是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。',
    system_message='你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。',
    roles=('<|im_start|>user\n', '<|im_start|>assistant\n'),
    sep_style=SeparatorStyle.MPT,
    sep='<|im_end|>',
    stop_str='<|endoftext|>',
)
register_conv_template(hermes_2)


internlm2_chat = Conversation(
    name='internlm2-chat',
    system_template='<|im_start|>system\n{system_message}',
    # note: The new system prompt was not used here to avoid changes in benchmark performance.
    # system_message='我是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。',
    system_message='你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。',
    roles=('<|im_start|>user\n', '<|im_start|>assistant\n'),
    sep_style=SeparatorStyle.MPT,
    sep='<|im_end|>',
)
register_conv_template(internlm2_chat)


phi3_chat = Conversation(
    name='phi3-chat',
    system_template='<|system|>\n{system_message}',
    # note: The new system prompt was not used here to avoid changes in benchmark performance.
    # system_message='我是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。',
    system_message='你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。',
    roles=('<|user|>\n', '<|assistant|>\n'),
    sep_style=SeparatorStyle.MPT,
    sep='<|end|>',
)
register_conv_template(phi3_chat)


internvl2_5 = Conversation(
    name='internvl2_5',
    system_template='<|im_start|>system\n{system_message}',
    system_message='你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。',
    roles=('<|im_start|>user\n', '<|im_start|>assistant\n'),
    sep_style=SeparatorStyle.MPT,
    sep='<|im_end|>\n',
)
register_conv_template(internvl2_5)


default_conversation = internvl2_5


def set_default_conv_template(conv_temp: Union[str, Conversation]):
    """Set default conversation template"""
    global default_conversation
    if not isinstance(conv_temp, (str, Conversation)):
        raise TypeError(f"type of param `conv_temp` should be str or Conversation")
    if isinstance(conv_temp, str):
        default_conversation = conv_templates[conv_temp]
    if isinstance(conv_temp, Conversation):
        default_conversation = conv_temp

    return default_conversation


if __name__ == "__main__":
    print(default_conversation.get_prompt())
