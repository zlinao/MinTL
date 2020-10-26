
import os
import math
import logging
from pprint import pformat
from argparse import ArgumentParser
from collections import defaultdict
from itertools import chain
import json

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, TensorDataset, Dataset
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Accuracy, Loss, MetricsLambda, RunningAverage
from ignite.contrib.handlers import ProgressBar, PiecewiseLinear
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler, OptimizerParamsHandler
from transformers import (AdamW, T5Tokenizer, T5ForConditionalGeneration, WEIGHTS_NAME,CONFIG_NAME)

import tensorflow.compat.v1 as tf
import numpy as np

ATTR_TO_SPECIAL_TOKEN = {'pad_token': '<pad>',
                         'additional_special_tokens': ["pricerange", "<go_r>", "<unk>", "<go_b>", "<go_a>", "<eos_u>", "<eos_r>", "<eos_b>"]} # some redundance

DATASET_PATHS = ["pretrain_data/preprocessed_data/schema.json", "pretrain_data/preprocessed_data/taskmaster_v1.json", "pretrain_data/preprocessed_data/taskmaster_v2.json"]
MODEL_INPUTS = ["input_ids", "masks", "context_ids", "context_masks", "target_ids", "target_inputs", "response_ids", "response_inputs"]

logger = logging.getLogger(__file__)


def random_spans_noise_mask(length=200,
                            noise_density=0.15,
                            mean_noise_span_length=3.0):
    """Noise mask consisting of random spans of noise tokens.
    The number of noise tokens and the number of noise spans and non-noise spans
    are determined deterministically as follows:
    num_noise_tokens = round(length * noise_density)
    num_nonnoise_spans = num_noise_spans = round(
    num_noise_tokens / mean_noise_span_length)
    Spans alternate between non-noise and noise, beginning with non-noise.
    Subject to the above restrictions, all masks are equally likely.
    Args:
    length: an int32 scalar (length of the incoming token sequence)
    noise_density: a float - approximate density of output mask
    mean_noise_span_length: a number
    Returns:
    a boolean tensor with shape [length]
    """
    orig_length = length
    # increase length to avoid degeneracy
    length = tf.maximum(length, 2)
    def to_int(x):
        return tf.cast(x, tf.int32)
    def to_float(x):
        return tf.cast(x, tf.float32)
    num_noise_tokens = to_int(tf.round(to_float(length) * noise_density))
    # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
    num_noise_tokens = tf.minimum(tf.maximum(num_noise_tokens, 1), length - 1)
    num_noise_spans = to_int(
        tf.round(to_float(num_noise_tokens) / mean_noise_span_length))
    # avoid degeneracy by ensuring positive number of noise spans
    num_noise_spans = tf.maximum(num_noise_spans, 1)
    num_nonnoise_tokens = length - num_noise_tokens
    # pick the lengths of the noise spans and the non-noise spans
    def _random_segmentation(num_items, num_segments):
        """Partition a sequence of items randomly into non-empty segments.
        Args:
        num_items: an integer scalar > 0
        num_segments: an integer scalar in [1, num_items]
        Returns:
        a Tensor with shape [num_segments] containing positive integers that add
        up to num_items
        """
        first_in_segment = tf.pad(
            tf.random.shuffle(to_int(tf.range(num_items - 1) < num_segments - 1),
                            seed=123),
            [[1, 0]])
        segment_id = tf.cumsum(first_in_segment)
        segment_length = tf.segment_sum(tf.ones_like(segment_id), segment_id)
        return segment_length
    noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans)
    nonnoise_span_lengths = _random_segmentation(
        num_nonnoise_tokens, num_noise_spans)
    interleaved_span_lengths = tf.reshape(
        tf.stack([nonnoise_span_lengths, noise_span_lengths], axis=1),
        [num_noise_spans * 2])
    span_starts = tf.cumsum(interleaved_span_lengths)[:-1]
    span_start_indicator = tf.unsorted_segment_sum(
        tf.ones_like(span_starts), span_starts, length)
    span_num = tf.cumsum(span_start_indicator)
    is_noise = tf.equal(span_num % 2, 1)
    return is_noise[:orig_length].numpy()


def average_distributed_scalar(scalar, args):
    """ Average a scalar over the nodes if we are in distributed training. We use this for distributed evaluation. """
    if args.local_rank == -1:
        return scalar
    scalar_t = torch.tensor(scalar, dtype=torch.float, device=args.device) / torch.distributed.get_world_size()
    torch.distributed.all_reduce(scalar_t, op=torch.distributed.ReduceOp.SUM)
    return scalar_t.item()

def add_special_tokens_(model, tokenizer):
    """ Add special tokens to the tokenizer and the model if they have not already been added. """
    orig_num_tokens = tokenizer.vocab_size
    num_added_tokens = tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN) # doesn't add if they are already there
    if num_added_tokens > 0:
        model.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_added_tokens)


class DatasetTrain(Dataset):
    """Custom data.Dataset compatible with DataLoader."""
    def __init__(self, data):
        self.data = data
        self.dataset_len = len(self.data)
        #self.max_len = max(len(x["input_ids"]) for x in self.data) 
    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        item = self.data[index]
        return item
    def __len__(self):
        return self.dataset_len


def padOutput(sequences, max_len=400, pad_token=0):
    lengths = [min(len(s),max_len) for s in sequences]
    num_samples = len(lengths)
    max_len = max(lengths)
    output_ids = np.ones((num_samples, max_len)) * (-100) #-100 ignore by cross entropy
    decoder_inputs = np.ones((num_samples, max_len)) * pad_token
    for idx, s in enumerate(sequences):
        trunc = s[:max_len]
        output_ids[idx, :lengths[idx]] = trunc
        decoder_inputs[idx, :lengths[idx]] = trunc
    return output_ids, decoder_inputs

def padInput(sequences, max_len=400, pad_token=0):
    lengths = [min(len(s),max_len) for s in sequences]
    num_samples = len(lengths)
    max_len = max(lengths)
    input_ids = np.ones((num_samples, max_len)) * pad_token
    masks = np.zeros((num_samples, max_len))

    for idx, s in enumerate(sequences):
        trunc = s[-max_len:]
        input_ids[idx, :lengths[idx]] = trunc
        masks[idx, :lengths[idx]] = 1
    return input_ids, masks


# def collate_fn(data):
#     batch = {"corrupted_context":[], "target":[], "response":[]}
#     padded_dataset = {}
#     # "input_ids":[], "input_masks":[] "target":[], 
#     for x in data:
#         corrupted_context = []
#         target = []
#         length = len(x["context_words"])
#         mask_bool = random_spans_noise_mask(length=length, noise_density=0.10, mean_noise_span_length=3.0)
#         mask_id = 0
#         for i in range(length):
#             if mask_bool[i]:
#                 if i>0 and mask_bool[i-1]:
#                     target.append(x["context_words"][i])
#                 else:
#                     target.append(f"<extra_id_{mask_id}>")
#                     corrupted_context.append(f"<extra_id_{mask_id}>")
#                     mask_id+=1
#             else:
#                 corrupted_context.append(x["context_words"][i])
#         target.append("<eos_b>")
#         batch["corrupted_context"].append(tokenizer.encode(" ".join(corrupted_context)))
#         batch["target"].append(tokenizer.encode(" ".join(target)))
#         batch["response"].append(tokenizer.encode(x["response"]))

#         print(tokenizer.decode(batch["corrupted_context"][-1]))
#         print(tokenizer.decode(batch["target"][-1]))
#         print(tokenizer.decode(batch["response"][-1]))

#     input_ids, masks = padInput(batch["corrupted_context"])
#     target_ids, target_inputs = padOutput(batch["target"])
#     response_ids, response_inputs = padOutput(batch["response"])
#     #inputs
#     padded_dataset["input_ids"] = torch.tensor(input_ids,dtype=torch.long)
#     padded_dataset["masks"] = torch.tensor(masks,dtype=torch.long)
#     padded_dataset["target_ids"] = torch.tensor(target_ids,dtype=torch.long)
#     padded_dataset["target_inputs"] = torch.tensor(target_inputs,dtype=torch.long)
#     padded_dataset["response_ids"] = torch.tensor(response_ids,dtype=torch.long)
#     padded_dataset["response_inputs"] = torch.tensor(response_inputs,dtype=torch.long)

#     return padded_dataset



def build_input_from_segments(history, reply):
    #mask_id0, eos_u, eos_r, eos_b = tokenizer.convert_tokens_to_ids(['<extra_id_0>', "<eos_u>", "<eos_r>", "<eos_b>"])
    #last one is user
    sequence = [s.split() + ["<eos_u>" if (len(history)-i) % 2 else "<eos_r>"] for i, s in enumerate(history)]
    instance = {}
    instance["context_words"] = list(chain(*sequence))
    instance["response"] = reply + " <eos_r>"
    return instance


def get_dataset():
    train = []
    valid = []
    for path in DATASET_PATHS:
        with open(path, encoding="utf-8") as f:
            dataset = json.load(f)

            train+=dataset[:-100]
            valid+=dataset[-100:]
    return train, valid

def get_data(args, tokenizer):
    """ Prepare the dataset for training and evaluation """
    train, valid = get_dataset()
    logger.info("Build inputs and labels")
    datasets = {"train": [], "valid": []}
    for dial in train:
        context = []
        for pair in dial:
            if len(pair)==2:
                context.append(pair[0])
                response = pair[1]
                instance = build_input_from_segments(context[-args.max_history:], response)
                datasets["train"].append(instance)
                context.append(pair[1])
    for dial in valid:
        context = []
        for pair in dial:
            if len(pair)==2:
                context.append(pair[0])
                response = pair[1]
                instance = build_input_from_segments(context[-args.max_history:], response)
                datasets["valid"].append(instance)
                context.append(pair[1])

    logger.info("Build train and validation dataloaders")
    train_dataset = DatasetTrain(datasets["train"])
    valid_dataset = DatasetTrain(datasets["valid"])
    #print(train_dataset.max_len, valid_dataset.max_len)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset) if args.distributed else None
    
    return train_dataset, valid_dataset, train_sampler, valid_sampler

def train():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="", help="Path or url of the dataset. If empty download from S3.")
    parser.add_argument("--model_checkpoint", type=str, default="t5-small", help="Path, url or short name of the model")
    parser.add_argument("--max_history", type=int, default=7, help="Number of previous exchanges to keep in history")
    parser.add_argument("--train_batch_size", type=int, default=10, help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int, default=10, help="Batch size for validation")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=12, help="Accumulate gradients on several steps")
    parser.add_argument("--lr", type=float, default=6e-4, help="Learning rate")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--n_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--fp16", type=str, default="", help="Set to O0, O1, O2 or O3 for fp16 training (see apex documentation)")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training (-1: not distributed)")
    parser.add_argument("--save_name", type=str, default="")
    parser.add_argument("--mask_ratio",type=float, default=0.15)
    parser.add_argument("--objective", type=str, default="span_denosing", help="response_generation, span_denosing, both")
    args = parser.parse_args()

    # logging is set to INFO (resp. WARN) for main (resp. auxiliary) process. logger.info => log main process only, logger.warning => log all processes
    logging.basicConfig(level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Running process %d", args.local_rank)  # This is a logger.warning: it will be printed by all distributed processes
    logger.info("Arguments: %s", pformat(args))

    # Initialize distributed training if needed
    args.distributed = (args.local_rank != -1)
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    logger.info("Prepare tokenizer, pretrained model and optimizer.")
    tokenizer = T5Tokenizer.from_pretrained(args.model_checkpoint)
    model = T5ForConditionalGeneration.from_pretrained(args.model_checkpoint)
    model.to(args.device)
    # Add special tokens if they are not already added
    add_special_tokens_(model, tokenizer)
    optimizer = AdamW(model.parameters(), lr=args.lr, correct_bias=True)

    # Prepare model for FP16 and distributed training if needed (order is important, distributed should be the last)
    if args.fp16:
        from apex import amp  # Apex is only required if we use fp16 training
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16)
    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)


    def collate_fn(data):
        batch = {"corrupted_context":[], "context":[], "target":[], "response":[]}
        padded_dataset = {}
        batch_size = len(data)
        resp_sos, context_sos = tokenizer.convert_tokens_to_ids(["<go_r>", "<go_b>",])
        for x in data:
            corrupted_context = ["fill : "]
            target = []
            length = len(x["context_words"])
            mask_bool = random_spans_noise_mask(length=length, noise_density=args.mask_ratio, mean_noise_span_length=3.0)
            mask_id = 0
            #print(mask_bool)
            for i in range(length):
                if mask_bool[i]:
                    if i>0 and mask_bool[i-1]:
                        target.append(x["context_words"][i])
                    else:
                        target.append(f"<extra_id_{mask_id}>")
                        target.append(x["context_words"][i])
                        corrupted_context.append(f"<extra_id_{mask_id}>")
                        mask_id+=1
                else:
                    corrupted_context.append(x["context_words"][i])
            target.append("<eos_b>")
            batch["context"].append(tokenizer.encode("response : " + " ".join(x["context_words"])))
            batch["corrupted_context"].append(tokenizer.encode(" ".join(corrupted_context)))
            batch["target"].append(tokenizer.encode(" ".join(target)))
            batch["response"].append(tokenizer.encode(x["response"]))
            # print(" ".join(x["context_words"]))
            # print(" ".join(corrupted_context))
            # print(" ".join(target))
            # print("")
            
            # print(tokenizer.decode(batch["corrupted_context"][-1]))
            # print(tokenizer.decode(batch["target"][-1]))
            # print(tokenizer.decode(batch["response"][-1]))
            # print("")
        context_ids, context_masks = padInput(batch["context"])
        input_ids, masks = padInput(batch["corrupted_context"])
        target_ids, target_inputs = padOutput(batch["target"])
        response_ids, response_inputs = padOutput(batch["response"])
        #inputs
        padded_dataset["input_ids"] = torch.tensor(input_ids,dtype=torch.long)
        padded_dataset["masks"] = torch.tensor(masks,dtype=torch.long)
        padded_dataset["context_ids"] = torch.tensor(context_ids,dtype=torch.long)
        padded_dataset["context_masks"] = torch.tensor(context_masks,dtype=torch.long)
        padded_dataset["target_ids"] = torch.tensor(target_ids,dtype=torch.long)
        padded_dataset["response_ids"] = torch.tensor(response_ids,dtype=torch.long)
        padded_dataset["target_inputs"] = torch.tensor(np.concatenate( (np.ones((batch_size,1))*context_sos  , target_inputs[:,:-1]), axis=1 ) ,dtype=torch.long)
        padded_dataset["response_inputs"] = torch.tensor( np.concatenate( ( np.ones((batch_size,1))*resp_sos, response_inputs[:,:-1]), axis=1 ) ,dtype=torch.long)

        return padded_dataset


    logger.info("Prepare datasets")
    train_dataset, valid_dataset, train_sampler, valid_sampler = get_data(args, tokenizer)

    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, shuffle=(not args.distributed), collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=args.valid_batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)

    logger.info("Train dataset length: {}".format(len(train_dataset)))
    logger.info("Valid dataset length: {}".format(len(valid_dataset)))
    # for batch in train_loader:
    #     #print(batch)
    #     exit(0)
    # Training function and trainer
    def update(engine, batch):
        model.train()
        batch = tuple(batch[input_name].to(args.device) for input_name in MODEL_INPUTS)
        input_ids, masks, context_ids, context_masks, target_ids, target_inputs, response_ids, response_inputs = batch
        # print("input")
        # print(tokenizer.decode(input_ids[0, :].tolist()))
        # print("context_ids")
        # print(tokenizer.decode(context_ids[0, :].tolist()))
        # print("target")
        # print(tokenizer.decode(target_ids[0, :].tolist()))
        # print("target In")
        # print(tokenizer.decode(target_inputs[0, :].tolist()))
        # print("response_ids")
        # print(tokenizer.decode(response_ids[0, :].tolist()))
        # print("response_inputs")
        # print(tokenizer.decode(response_inputs[0, :].tolist()))
        #exit(0)
        outputs = model(
            input_ids, attention_mask=masks, decoder_input_ids=target_inputs, lm_labels=target_ids
        )
        context_loss = outputs[0]
        
        outputs = model(context_ids, 
                        attention_mask=context_masks,
                        decoder_input_ids=response_inputs,
                        lm_labels=response_ids
                        )

        resp_loss = outputs[0]

        loss = (context_loss + resp_loss) / args.gradient_accumulation_steps

        loss = (context_loss) / args.gradient_accumulation_steps
        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_norm)
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        if engine.state.iteration % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        return loss.item()
    trainer = Engine(update)

    # Evaluation function and evaluator (evaluator output is the input of the metrics)
    def inference(engine, batch):
        model.eval()
        with torch.no_grad():
            batch = tuple(batch[input_name].to(args.device) for input_name in MODEL_INPUTS)
            input_ids, masks, context_ids, context_masks, target_ids, target_inputs, response_ids, response_inputs = batch

            outputs = model(
                input_ids, attention_mask=masks, decoder_input_ids=target_inputs#, lm_labels=target_ids
            )
            
            context_logits = outputs[0]
            outputs = model(context_ids,
                            attention_mask=context_masks,
                            decoder_input_ids=response_inputs,
                            #lm_labels=response_ids
                            )
            resp_logits = outputs[0]

            context_logits_flat_shifted = context_logits.view(-1, context_logits.size(-1))
            context_labels_flat_shifted = target_ids.view(-1)

            resp_logits_flat_shifted = resp_logits.view(-1, resp_logits.size(-1))
            resp_labels_flat_shifted = response_ids.view(-1)

            return (context_logits_flat_shifted, resp_logits_flat_shifted), (context_labels_flat_shifted, resp_labels_flat_shifted)
            #return (context_logits_flat_shifted, context_logits_flat_shifted), (context_labels_flat_shifted, context_labels_flat_shifted)
    evaluator = Engine(inference)

    # Attach evaluation to trainer: we evaluate when we start the training and at the end of each epoch
    trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda _: evaluator.run(val_loader))
    if args.n_epochs < 1:
        trainer.add_event_handler(Events.COMPLETED, lambda _: evaluator.run(val_loader))
    # if args.eval_before_start:
    #     trainer.add_event_handler(Events.STARTED, lambda _: evaluator.run(val_loader))

    # Make sure distributed data samplers split the dataset nicely between the distributed processes
    if args.distributed:
        trainer.add_event_handler(Events.EPOCH_STARTED, lambda engine: train_sampler.set_epoch(engine.state.epoch))
        evaluator.add_event_handler(Events.EPOCH_STARTED, lambda engine: valid_sampler.set_epoch(engine.state.epoch))

    # Linearly decrease the learning rate from lr to zero
    scheduler = PiecewiseLinear(optimizer, "lr", [(0, args.lr), (args.n_epochs * len(train_loader), 0.0)])
    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

    # Prepare metrics - note how we compute distributed metrics
    RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")
    metrics = {"span": Loss(torch.nn.CrossEntropyLoss(ignore_index=-100), output_transform=lambda x: (x[0][0], x[1][0])),
               "response": Loss(torch.nn.CrossEntropyLoss(ignore_index=-100), output_transform=lambda x: (x[0][1], x[1][1]))}
    metrics.update({"average_span": MetricsLambda(average_distributed_scalar, metrics["span"], args),
                    "average_response": MetricsLambda(average_distributed_scalar, metrics["response"], args)})
    metrics["average_response"] = MetricsLambda(math.exp, metrics["average_response"])
    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    # On the main process: add progress bar, tensorboard, checkpoints and save model, configuration and tokenizer before we start to train
    if args.local_rank in [-1, 0]:
        pbar = ProgressBar(persist=True)
        pbar.attach(trainer, metric_names=["loss"])
        evaluator.add_event_handler(Events.COMPLETED, lambda _: pbar.log_message("Validation: %s" % pformat(evaluator.state.metrics)))

        if not os.path.exists(f"pretrained_model/{args.save_name}"):
            os.makedirs(f"pretrained_model/{args.save_name}")
        log_dir = f"pretrained_model/{args.save_name}"
        tb_logger = TensorboardLogger(log_dir)

        tb_logger.attach(trainer, log_handler=OutputHandler(tag="training", metric_names=["loss"]), event_name=Events.ITERATION_COMPLETED)
        tb_logger.attach(trainer, log_handler=OptimizerParamsHandler(optimizer), event_name=Events.ITERATION_STARTED)
        tb_logger.attach(evaluator, log_handler=OutputHandler(tag="validation", metric_names=list(metrics.keys()), another_engine=trainer), event_name=Events.EPOCH_COMPLETED)

        checkpoint_handler = ModelCheckpoint(log_dir, 'checkpoint', save_interval=1, n_saved=3)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {'mymodel': getattr(model, 'module', model)})  # "getattr" takes care of distributed encapsulation

        torch.save(args, log_dir + '/model_training_args.bin')
        getattr(model, 'module', model).config.to_json_file(os.path.join(log_dir, CONFIG_NAME))
        tokenizer.save_pretrained(log_dir)

    # Run the training
    trainer.run(train_loader, max_epochs=args.n_epochs)

    # On the main process: close tensorboard logger and rename the last checkpoint (for easy re-loading with OpenAIGPTModel.from_pretrained method)
    if args.local_rank in [-1, 0] and args.n_epochs > 0:
        os.rename(os.path.join(log_dir, checkpoint_handler._saved[-1][1]), os.path.join(log_dir, WEIGHTS_NAME))  # TODO: PR in ignite to have better access to saved file paths (cleaner)
        tb_logger.close()

if __name__ == "__main__":
    train()
