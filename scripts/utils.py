import ast
import logging
import os
from torch.nn import functional as F
import subprocess
import sys

from peft import PeftModel
from yamlize import Object, Attribute, Sequence, StrList, Typed
import torch
import transformers
from transformers import AutoTokenizer, CodeGenForCausalLM, AutoModelForCausalLM

logger = logging.getLogger()

def set_logging(args, log_file):
    handlers = []
    handlers.append(logging.StreamHandler(stream=sys.stdout))
    if log_file is not None:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
        handlers=handlers
    )
    args.logger = logger

def set_devices(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    args.logger.info('Device: %s, n_gpu: %s', device, args.n_gpu)

def parallelize_model(model, args):
    if args.n_gpu > 1:
        model.parallelize()
        input_device = model.transformer.first_device
    else:
        model.to(args.device)
        input_device = args.device
    return input_device

def load_model(model_type, path, is_training, args):

    transformers.logging.set_verbosity_error()
    tokenizer = AutoTokenizer.from_pretrained(path)
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token_id = tokenizer.bos_token_id
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if 'codegen' not in args.model_name_or_path:
        if model_type == 'lm':
            model = AutoModelForCausalLM.from_pretrained(path)
        else:
            base_model = AutoModelForCausalLM.from_pretrained(path)
            model = PeftModel.from_pretrained(base_model, args.peft_model)

    else:
        if model_type == 'lm':
            model = CodeGenForCausalLM.from_pretrained(path)
        else:
            base_model = CodeGenForCausalLM.from_pretrained(path)
            model =PeftModel.from_pretrained(base_model, args.peft_model)

    model.resize_token_embeddings(len(tokenizer))
    input_device = parallelize_model(model, args)
    return tokenizer, model, input_device

def try_parse(code, lang):
    if lang == 'py':
        try:
            ast.parse(code)
            return 0
        except:
            return 1
    elif lang == 'c':
        cmd = 'gcc -c -x c -'
        process = subprocess.run(cmd, shell=True, timeout=5, input=code.encode(), stderr=subprocess.DEVNULL)
        if process.returncode == 0:
            return 0
        else:
            return 1
    else:
        raise NotImplementedError()



class Problem(Object):
    '''
        yamlize 是一个用于在 Python 中操作 YAML 数据的库。YAML 是一种用于表示配置文件和数据的文本格式，通常易读且易于编写。

    '''
    name = Attribute(type=str)
    language = Attribute(type=str)
    prompt = Attribute(type=str)
    tests = Attribute(type=str)
    completions = Attribute(type=StrList)
    stop_tokens = Attribute(type=StrList)

def add_to_loss_dict(acc_loss_dict, loss_dict):
    for key, val in loss_dict.items():
        if key not in acc_loss_dict:
            acc_loss_dict[key] = 0.0
        acc_loss_dict[key] += val

def report_loss_dict(loss_dict, steps):
    ss = []
    for key, val in loss_dict.items():
        if key == 'kl_loss':
            r = 8
        else:
            r = 4
        ss.append(f'{key}: {round(val/steps, r)}')
    return ', '.join(ss)

def save_model(model, path, args):
    model.save_pretrained(path)


def save(path, model, tokenizer, step, epoch, optimizer, scheduler, args):
    if not os.path.exists(path):
        os.makedirs(path)
    save_model(model, path, args)
    tokenizer.save_pretrained(path)
    step_file = os.path.join(path, 'step_file.txt')
    with open(step_file, 'w') as f:
        f.write(str(step)+'\n')
    epoch_file = os.path.join(path, 'epoch_file.txt')
    with open(epoch_file, 'w') as f:
        f.write(str(epoch)+'\n')
    if optimizer is not None:
        torch.save(optimizer.state_dict(), os.path.join(path, 'optimizer.pt'))
    if scheduler is not None:
        torch.save(scheduler.state_dict(), os.path.join(path, 'scheduler.pt'))


def sample(probs : torch.Tensor, num_samples: int = 1):
    idx_next = torch.multinomial(probs, num_samples=num_samples)
    if (idx_next.item() == 0):
        raise RuntimeError
    return idx_next

def norm_logits(logits : torch.Tensor, temperature : float, top_k : float, top_p : float) -> torch.Tensor:
    """

    Args:
        logits (torch.Tensor): shape (1, vocab)
        temperature (float): temperature
        top_k (float): top_k
        top_p (float): top_p

    Returns:
        torch.Tensor: next token with shape as (batch,  1)
    """
    assert logits.dim() == 2
    logits = logits / temperature
    logits = top_k_top_p_filter(logits, top_k=top_k, top_p=top_p)
    probs = F.softmax(logits, dim=1)
    return probs

def top_k_top_p_filter(logits: torch.Tensor, top_k: int = 0, top_p: float = 0.0):
    """

    Args:
        logits (torch.Tensorpe_): 2D tensor with shape (batch, vocab)
        top_k (int, optional): top_k. Defaults to 0.
        top_p (float, optional): top_p. Defaults to 0.0.

    Returns:
        torch.Tensor: a renormalized logits
    """
    if top_k > 0:
        filter = torch.topk(logits, min(top_k, logits.size(-1)))[0]
        logits[logits < filter[:, [-1]]] = float('-inf')
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(
            F.softmax(sorted_logits, dim=-1), dim=-1)
        filter = cumulative_probs > top_p
        filter[..., 1:] = filter[..., :-1].clone()
        filter[..., 0] = 0
        indices_to_remove = filter.scatter(1, sorted_indices, filter)
        logits[indices_to_remove] = float('-inf')
    return logits

def max_fn(x):
    """
        norm(max (x, 0))
    """
    x_max = torch.where(x > 0, x, torch.zeros_like(x))
    x_max_sum = torch.sum(x_max, dim=1, keepdim=True)
    return x_max / x_max_sum

def _split_model_outputs(outputs, new_outputs, cur_len, added_len, is_decoder_attention=False):
    """
    Given the (decoder/cross attentions)/(decoder hidden states) for multiple generated tokens, splits it into a tuple
    where each member corresponds to a single generated token.
    """
    # Retrocompatibility: in our generation functions, the first iteration includes the attention/hidden states for the
    # prompt.
    if len(outputs) == 0:
        new_tuple = ()
        for layer in new_outputs:
            last_dim_size = cur_len if is_decoder_attention else layer.shape[-1]
            new_tuple += (layer[..., :cur_len, :last_dim_size],)
        outputs += (new_tuple,)
        # The first iteration contains the prompt + 1 generated token, let's update the length variables accordingly
        cur_len += 1
        added_len -= cur_len

    for i in range(added_len):
        new_tuple = ()
        for layer in new_outputs:
            last_dim_size = cur_len + i if is_decoder_attention else layer.shape[-1]
            new_tuple += (layer[..., i: i + 1, :last_dim_size],)
        outputs += (new_tuple,)
    return outputs