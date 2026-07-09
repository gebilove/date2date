"""训练、生成、评估函数。"""

import torch

from data import (
    tensor_from_string,
    string_from_tensor,
    SOS_token,
    EOS_token,
    vocab_size,
)


def train_one_sample(input_str, target_str, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    input_tensor = tensor_from_string(input_str)
    target_tensor = tensor_from_string(target_str, True)
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    encoder_output, encoder_hidden = encoder(input_tensor)
    decoder_outputs, decoder_hidden = decoder(encoder_output, target_tensor)
    logits = decoder_outputs.reshape(-1, vocab_size)
    target = target_tensor.reshape(-1)
    loss = criterion(logits, target)
    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()
    return loss.item()


def generate(input_str, encoder, decoder, max_len=20):
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        input_tensor = tensor_from_string(input_str)
        encoder_output, encoder_hidden = encoder(input_tensor)
        decoder_outputs, decoder_hidden = decoder(encoder_output, None, max_len)
        # decoder_outputs.shape (batch,seq_len,vocab_size)
        pred_tokens = decoder_outputs.argmax(dim=2)
        outputs = []
        for token in pred_tokens[0]:
            if token.item() == EOS_token:
                break
            outputs.append(token)
        if len(outputs) == 0:
            return ""


    return string_from_tensor(torch.stack(outputs).unsqueeze(0))


def eval_samples(samples, encoder, decoder, verbose=False):
    encoder.eval()
    decoder.eval()

    correct_count = 0

    for input_str, target_str in samples:
        pred = generate(input_str, encoder, decoder)
        correct = pred == target_str

        if correct:
            correct_count += 1

        if verbose:
            print("input:  ", input_str)
            print("pred:   ", pred)
            print("target: ", target_str)
            print("correct:", correct)
            print("---")

    total = len(samples)
    acc = correct_count / total

    return correct_count, total, acc


def char_accuracy(samples, encoder, decoder):
    total_chars = 0
    correct_chars = 0

    for input_str, target_str in samples:
        pred = generate(input_str, encoder, decoder, max_len=20)
        max_compare_len = min(len(pred), len(target_str))

        for i in range(max_compare_len):
            total_chars += 1
            if pred[i] == target_str[i]:
                correct_chars += 1

        total_chars += max(0, len(target_str) - max_compare_len)

    return correct_chars / total_chars
