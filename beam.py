import einops


@dataclass
class Beams:
    """Class to store beams during beam search."""

    model: DemoTransformer
    tokenizer: GPT2TokenizerFast
    logprob_sums: Float[Tensor, "batch"]
    tokens: Int[Tensor, "batch seq"]

    def __getitem__(self, batch_idx) -> "Beams":
        """Allows you to create new beams from old beams by slicing along batch dim (useful for `filter`)."""
        return Beams(
            self.model,
            self.tokenizer,
            self.logprob_sums[batch_idx],
            self.tokens[batch_idx],
        )

    @property
    def logprobs_and_completions(self) -> list[tuple[float, str]]:
        """Returns self as a list of logprob sums and completions (useful for getting final output)."""
        return [
            (logprob_sum.item(), self.tokenizer.decode(tokens))
            for (logprob_sum, tokens) in zip(self.logprob_sums, self.tokens)
        ]

    def generate(self, k: int, no_repeat_ngram_size: int | None = None) -> "Beams":
        """
        Starting from the current set of beams (i.e. self.tokens) and returns a new set of `len(self.tokens) * k` beams,
        containing the best `k` continuations for each of the original beams.

        Optional argument `no_repeat_ngram_size` means your model won't generate any sequences with a repeating n-gram
        of this length.
        """
        # Get the output logprobs for the next token (for every sequence in current beams)
        logprobs = self.model(self.tokens)[:, -1, :].log_softmax(-1)

        # Get the top `toks_per_beam` tokens for each sequence
        topk_logprobs, topk_tokenIDs = self.get_topk_non_repeating(
            logprobs, no_repeat_ngram_size, k=k
        )

        # Add new logprobs & concat new tokens. When doing this, we need to add an extra `k` dimension since our current
        # logprobs & tokens have shape (batch,) and (batch, seq), but our new ones both have shape (batch, k)
        new_logprob_sums = (
            einops.repeat(self.logprob_sums, "b -> b k", k=k) + topk_logprobs
        )
        new_tokens = t.concat(
            [
                einops.repeat(self.tokens, "b s -> b k s", k=k),
                topk_tokenIDs.unsqueeze(-1),
            ],
            dim=-1,
        )

        return Beams(
            self.model,
            self.tokenizer,
            new_logprob_sums.flatten(),
            new_tokens.flatten(0, 1),
        )

    def filter(self, k: int) -> tuple["Beams", "Beams"]:
        """
        Returns:
            best_beams: Beams
                filtered version of self, containing all best `k` which are also not terminated.
            early_terminations: Beams
                filtered version of self, containing all best `k` which are also terminated.
        """
        # Get the indices of top `k` beams
        top_beam_indices = self.logprob_sums.topk(k=k, dim=0).indices.tolist()
        # Get the indices of terminated sequences
        new_tokens = self.tokens[:, -1]
        terminated_indices = t.nonzero(new_tokens == self.tokenizer.eos_token_id)

        # Get the indices of the `k` best sequences (some terminated, some not terminated)
        best_continuing = [i for i in top_beam_indices if i not in terminated_indices]
        best_terminated = [i for i in top_beam_indices if i in terminated_indices]

        # Return the beam objects from these indices
        return self[best_continuing], self[best_terminated]

    def get_topk_non_repeating(
        self,
        logprobs: Float[Tensor, "batch d_vocab"],
        no_repeat_ngram_size: int | None,
        k: int,
    ) -> tuple[Float[Tensor, "k"], Int[Tensor, "k"]]:
        """
        logprobs:
            tensor of the log-probs for the next token
        no_repeat_ngram_size:
            size of ngram to avoid repeating
        k:
            number of top logits to return, for each beam in our collection

        Returns:
            equivalent to the output of `logprobs.topk(dim=-1)`, but makes sure that no returned tokens would produce an
            ngram of size `no_repeat_ngram_size` which has already appeared in `self.tokens`.
        """
        batch, seq_len = self.tokens.shape

        # If completion isn't long enough for a repetition, or we have no restructions, just return topk
        if (no_repeat_ngram_size is not None) and (seq_len > no_repeat_ngram_size - 1):
            # Otherwise, we need to check for ngram repetitions
            # First, get the most recent `no_repeat_ngram_size-1` tokens
            last_ngram_prefix = self.tokens[:, seq_len - (no_repeat_ngram_size - 1) :]
            # Next, find all the tokens we're not allowed to generate, by checking all past ngrams for a match
            for i in range(seq_len - (no_repeat_ngram_size - 1)):
                ngrams = self.tokens[:, i : i + no_repeat_ngram_size]  # (batch, ngram)
                ngrams_are_repeated = (ngrams[:, :-1] == last_ngram_prefix).all(
                    -1
                )  # (batch,)
                ngram_end_tokens = ngrams[:, [-1]]  # (batch, 1)
                # Fill logprobs with neginf wherever the ngrams are repeated
                logprobs[range(batch), ngram_end_tokens] = t.where(
                    ngrams_are_repeated,
                    -1.0e10,
                    logprobs[range(batch), ngram_end_tokens],
                )

        # Finally, get our actual tokens
        return logprobs.topk(k=k, dim=-1)

    def print(self, title="Best completions", max_print_chars=80) -> None:
        """
        Prints out a set of sequences with their corresponding logprob sums.
        """
        if len(self.tokens) == 0:
            return
        table = Table("logprob sum", "completion", title=title)
        for logprob_sum, tokens in zip(self.logprob_sums, self.tokens):
            text = self.tokenizer.decode(tokens)
            if len(repr(text)) > max_print_chars:
                text = (
                    text[: int(0.3 * max_print_chars)]
                    + " ... "
                    + text[-int(0.7 * max_print_chars) :]
                )
            table.add_row(f"{logprob_sum:>8.3f}", repr(text))
        rprint(table)


@t.inference_mode()
def beam_search(
    self: TransformerSampler,
    prompt: str,
    num_return_sequences: int,
    num_beams: int,
    max_new_tokens: int,
    no_repeat_ngram_size: int | None = None,
) -> list[tuple[float, str]]:
    """
    Implements a beam search, by repeatedly performing the `generate` and `filter` steps (starting from the initial
    prompt) until either of the two stopping criteria are met: (1) we've generated `max_new_tokens` tokens, or (2)
    we've generated `num_returns_sequences` terminating sequences.
    """
    assert num_return_sequences <= num_beams
    self.model.eval()

    tokens = self.tokenizer.encode(prompt, return_tensors="pt").to(device)

    final_logprobs_and_completions = (
        []
    )  # we add to this list as we get terminated beams
    best_beams = Beams(
        self.model, self.tokenizer, t.tensor([0.0]).to(device), tokens
    )  # start with just 1 beam

    for _ in tqdm(range(max_new_tokens)):
        t.cuda.empty_cache()

        # Generate & filter beams
        best_beams = best_beams.generate(
            k=num_beams, no_repeat_ngram_size=no_repeat_ngram_size
        )
        best_beams, best_beams_terminated = best_beams.filter(k=num_beams)

        # Add terminated beams to our list, and return early if we have enough
        final_logprobs_and_completions.extend(
            best_beams_terminated.logprobs_and_completions
        )
        if len(final_logprobs_and_completions) >= num_return_sequences:
            return final_logprobs_and_completions[:num_return_sequences]

    # Return terminated beams plus the best ongoing beams of length `orig_len + max_new_tokens`
    final_logprobs_and_completions.extend(best_beams.logprobs_and_completions)
    return final_logprobs_and_completions[:num_return_sequences]


TransformerSampler.beam_search = beam_search
