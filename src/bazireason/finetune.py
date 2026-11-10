"""Training-data tokenization for local LoRA fine-tuning with MLX-LM.

Each row is ``{"messages": [...]}`` ending with one assistant message. The
prompt part is rendered exactly as at inference time (generation prompt added,
thinking disabled), and the completion is the assistant text followed by the
end-of-turn token. The loss is computed on the completion only, so the model is
trained on the same token boundary it is later asked to continue from.
"""


class PromptCompletionDataset:
    """Rows of (token ids, prompt length) for ``mlx_lm.tuner.trainer``."""

    def __init__(self, rows, tokenizer, enable_thinking=False):
        self._rows = rows
        self.tokenizer = tokenizer
        self.enable_thinking = enable_thinking
        self.end_of_turn = tokenizer.eos_token

    def process(self, row):
        messages = row["messages"]
        if messages[-1]["role"] != "assistant":
            raise ValueError("A training row must end with an assistant message")
        prompt = list(
            self.tokenizer.apply_chat_template(
                messages[:-1],
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
                return_dict=False,
            )
        )
        completion = self.tokenizer.encode(
            messages[-1]["content"] + self.end_of_turn, add_special_tokens=False
        )
        return (prompt + list(completion), len(prompt))

    def __getitem__(self, index):
        return self._rows[index]

    def __len__(self):
        return len(self._rows)
