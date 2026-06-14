"""
Monkey-patch fixes for TRL GRPOTrainer multimodal alignment with Qwen3-VL.

Fix 1: Rebuild mm_token_type_ids from input_ids to fix padding direction mismatch.
Fix 2: Strip generated image/video pad tokens from completion_ids to prevent
       orphan image tokens without matching pixel_values features.
"""

import re as _re

import torch


def apply_grpo_fixes(trainer_class):
    """Apply all monkey-patches to GRPOTrainer.

    Idempotent: patches are applied at most once per class to avoid nested
    wrappers and accidental in-place modifications stacking across rounds.
    """
    attr = "_grpo_fixes_applied"
    if getattr(trainer_class, attr, False):
        return
    _patch_prepare_inputs(trainer_class)
    _patch_get_per_token_logps_and_entropies(trainer_class)
    _patch_log_completions(trainer_class)
    setattr(trainer_class, attr, True)


def _patch_prepare_inputs(trainer_class):
    """Fix 1: Rebuild mm_token_type_ids / token_type_ids from actual input_ids
    so padding direction matches the attention_mask.
    """
    orig = trainer_class._prepare_inputs

    def _prepare_inputs(self, inputs):
        # Call original
        result = orig(self, inputs)

        # --- Rebuild mm_token_type_ids / token_type_ids ONLY if missing or wrong shape ---
        # TRL 1.5.1 _generate_and_score_completions already extends these correctly
        # with zeros for the completion part. We only fix them if the shape is wrong.
        prompt_ids = result.get("prompt_ids")
        completion_ids = result.get("completion_ids")
        if prompt_ids is None:
            return result

        # Full sequence length = prompt + completion
        full_len = prompt_ids.shape[-1]
        if completion_ids is not None:
            full_len += completion_ids.shape[-1]

        if self._is_vlm:
            # Only rebuild mm_token_type_ids if missing or shape mismatch
            mm_ids_existing = result.get("mm_token_type_ids")
            if mm_ids_existing is None or mm_ids_existing.shape[-1] != full_len:
                # Build from full prompt+completion ids if available
                if completion_ids is not None:
                    input_ids = torch.cat([prompt_ids, completion_ids], dim=-1)
                else:
                    input_ids = prompt_ids
                mm_ids = self._rebuild_mm_ids_from_input_ids(input_ids)
                if mm_ids is not None:
                    result["mm_token_type_ids"] = mm_ids

            # Only rebuild token_type_ids if missing or shape mismatch
            token_type_ids_existing = result.get("token_type_ids")
            if token_type_ids_existing is None or token_type_ids_existing.shape[-1] != full_len:
                if completion_ids is not None:
                    input_ids = torch.cat([prompt_ids, completion_ids], dim=-1)
                else:
                    input_ids = prompt_ids
                token_type_ids = self._rebuild_token_type_ids(input_ids)
                if token_type_ids is not None:
                    result["token_type_ids"] = token_type_ids

        return result

    trainer_class._prepare_inputs = _prepare_inputs

    # --- Helper: rebuild mm_token_type_ids from input_ids ---
    def _rebuild_mm_ids_from_input_ids(self, input_ids):
        if not self._is_vlm or self._image_pad_token_id is None:
            return None
        mm_ids = torch.zeros_like(input_ids)
        if self._image_pad_token_id is not None:
            mm_ids[input_ids == self._image_pad_token_id] = 1
        if self._video_pad_token_id is not None:
            mm_ids[input_ids == self._video_pad_token_id] = 2
        return mm_ids

    trainer_class._rebuild_mm_ids_from_input_ids = _rebuild_mm_ids_from_input_ids

    # --- Helper: rebuild token_type_ids for Qwen3VL ---
    def _rebuild_token_type_ids(self, input_ids):
        if not self._is_vlm or self._image_pad_token_id is None:
            return None
        token_type_ids = torch.zeros_like(input_ids)
        if self._image_pad_token_id is not None:
            token_type_ids[input_ids == self._image_pad_token_id] = 1
        if self._video_pad_token_id is not None:
            token_type_ids[input_ids == self._video_pad_token_id] = 2
        # Note: audio pad token id not stored in TRL, so 3 is never set here.
        return token_type_ids

    trainer_class._rebuild_token_type_ids = _rebuild_token_type_ids


def _patch_get_per_token_logps_and_entropies(trainer_class):
    """Fix 2: Strip image/video pad tokens from completion_ids so
    pixel_values and image_grid_thw match exactly the text tokens.
    """
    orig = trainer_class._get_per_token_logps_and_entropies

    def _get_per_token_logps_and_entropies(
        self,
        model,
        input_ids,
        attention_mask,
        logits_to_keep,
        batch_size=None,
        compute_entropy=False,
        pixel_values=None,
        image_grid_thw=None,
        num_images=None,
        pixel_attention_mask=None,
        image_sizes=None,
        token_type_ids=None,
        mm_token_type_ids=None,
        image_position_ids=None,
    ):
        # Strip orphan image/video tokens from the completion part.
        # This prevents "Image features and image tokens do not match".
        if self._is_vlm and hasattr(self, "_image_pad_token_id"):
            if input_ids is not None and input_ids.numel() > 0:
                # Only strip the completion portion (not prompt)
                # input_ids is (prompt + completion); completion is the last logits_to_keep tokens
                seq_len = input_ids.shape[-1]
                prompt_len = seq_len - logits_to_keep

                completion_part = input_ids[:, prompt_len:]
                if completion_part.numel() > 0:
                    # Find first image/video pad in completion
                    image_mask = (
                        completion_part == self._image_pad_token_id
                        if self._image_pad_token_id is not None
                        else torch.zeros_like(completion_part, dtype=torch.bool)
                    )
                    video_mask = (
                        completion_part == self._video_pad_token_id
                        if self._video_pad_token_id is not None
                        else torch.zeros_like(completion_part, dtype=torch.bool)
                    )
                    bad_mask = image_mask | video_mask
                    if bad_mask.any():
                        # Clone before modifying so we don't mutate the caller's tensor
                        # (prompt_completion_ids is reused for policy/ref forwards).
                        input_ids = input_ids.clone()
                        completion_part = input_ids[:, prompt_len:]
                        # Truncate each row at the first bad token
                        for b in range(completion_part.shape[0]):
                            bad_positions = bad_mask[b].nonzero(as_tuple=True)[0]
                            if len(bad_positions) > 0:
                                first_bad = bad_positions[0].item()
                                # Truncate completion at first bad token
                                input_ids[b, prompt_len + first_bad :] = 0

        return orig(
            self,
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            batch_size=batch_size,
            compute_entropy=compute_entropy,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            num_images=num_images,
            pixel_attention_mask=pixel_attention_mask,
            image_sizes=image_sizes,
            token_type_ids=token_type_ids,
            mm_token_type_ids=mm_token_type_ids,
            image_position_ids=image_position_ids,
        )

    trainer_class._get_per_token_logps_and_entropies = _get_per_token_logps_and_entropies


def _patch_log_completions(trainer_class):
    """Fix 3: Log first completion every 5 steps to verify model behavior."""
    orig_generate_and_score = trainer_class._generate_and_score_completions

    def _generate_and_score_completions(self, inputs):
        result = orig_generate_and_score(self, inputs)
        # Every 5 steps: log first completion
        if self.state.global_step % 5 == 0 and "completion" in self._logs and self._logs["completion"]:
            first_completion = self._logs["completion"][0]
            if isinstance(first_completion, list) and len(first_completion) > 0:
                text = first_completion[0].get("content", str(first_completion[0]))
            else:
                text = str(first_completion)
            text = _re.sub(r"\n{2,}", " [...] ", text)
            text = text.replace("\n", " | ")
            print(f"\n[Step {self.state.global_step}] First completion: {text[:400]}\n")
        return result

    trainer_class._generate_and_score_completions = _generate_and_score_completions
