import argparse
import csv
import os
import re
from ast import literal_eval

import numpy as np
import pandas as pd
import torch
from seqeval.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import DistilBertModel, DistilBertPreTrainedModel, DistilBertTokenizerFast, Trainer, TrainingArguments
from transformers.modeling_outputs import TokenClassifierOutput


def main(
        file_path=None,
        do_train=None,
        do_eval=None,
        do_test=None,
        model_name_or_path=None,
        tokenizer_name_or_path=None,
        model_output_dir=None,
        result_output_dir=None,
        tfidf=None
):
    def read_tsd(file_path):
        df = pd.read_csv(file_path)
        df['spans'] = df.spans.apply(literal_eval)

        token_docs = []
        tag_docs = []

        for _, row in df.iterrows():
            tokens = []
            tags = []

            text = re.findall(r"\w+(?:'\w+)*|[^\w]", row['text'].encode('ascii', 'ignore').decode('ascii'))
            offset = 0

            for token in text:
                length = len(token)
                if token.isspace() or token == chr(127):
                    offset += length
                    continue

                tokens.append(token)

                toxic = False
                for i in range(len(token)):
                    if i + offset in row['spans']:
                        toxic = True
                if toxic:
                    tags.append('B-toxic') if tags == [] or tags[-1] == 'O' else tags.append('I-toxic')
                else:
                    tags.append('O')

                offset += length

            token_docs.append(tokens)
            tag_docs.append(tags)

        return token_docs, tag_docs

    texts, tags = read_tsd(file_path)

    unique_tags = ['B-toxic', 'I-toxic', 'O']
    tag2id = {tag: id for id, tag in enumerate(unique_tags)}
    id2tag = {id: tag for tag, id in tag2id.items()}

    tokenizer = DistilBertTokenizerFast.from_pretrained(tokenizer_name_or_path)
    encodings = tokenizer(texts, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True)

    def encode_tags(tags, encodings):
        labels = [[tag2id[tag] for tag in doc] for doc in tags]
        encoded_labels = []
        for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):
            doc_enc_labels = np.ones(len(doc_offset), dtype=int) * -100
            arr_offset = np.array(doc_offset)

            doc_enc_labels[(arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)] = doc_labels
            encoded_labels.append(doc_enc_labels.tolist())

        return encoded_labels

    labels = encode_tags(tags, encodings)

    class TSDDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    encodings.pop('offset_mapping')
    dataset = TSDDataset(encodings, labels)

    if tfidf:
        def identity_tokenizer(text):
            return text

        vectorizer = TfidfVectorizer(lowercase=False, tokenizer=identity_tokenizer)
        corpus = [tokenizer.tokenize(' '.join(tokens)) for tokens in texts]

        X = vectorizer.fit_transform(corpus)
        feature_names = vectorizer.get_feature_names()

    class TSDModel(DistilBertPreTrainedModel):
        def __init__(self, config):
            super().__init__(config)
            self.num_labels = config.num_labels

            self.distilbert = DistilBertModel(config)
            self.dropout = torch.nn.Dropout(config.dropout)
            self.classifier = torch.nn.Linear(config.hidden_size + tfidf, config.num_labels)

            self.init_weights()

        def forward(
                self,
                input_ids=None,
                attention_mask=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
        ):
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            outputs = self.distilbert(
                input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            sequence_output = outputs[0]
            sequence_output = self.dropout(sequence_output)

            if tfidf:
                shape = list(sequence_output.shape)
                shape[-1] += 1
                new_sequence_output = torch.zeros(shape)

                for i, doc in enumerate(sequence_output):
                    feature_index = X[i, :].nonzero()[1]
                    tfidf_scores = zip(feature_index, [X[i, x] for x in feature_index])
                    words, scores = map(list, zip(*[(feature_names[k], score) for (k, score) in tfidf_scores]))

                    for j, token in enumerate(doc):
                        if j == 0:
                            new_sequence_output[i][j] = torch.cat((token, torch.tensor([0])))
                            continue

                        for word, score in zip(words, scores):
                            try:
                                if word == corpus[i][j - 1]:
                                    new_sequence_output[i][j] = torch.cat((token, torch.tensor([score])))
                                    break
                            except IndexError:
                                new_sequence_output[i][j] = torch.cat((token, torch.tensor([0])))
                                break

                sequence_output = new_sequence_output

            logits = self.classifier(sequence_output)

            loss = None
            if labels is not None:
                loss_fct = torch.nn.CrossEntropyLoss()
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = logits.view(-1, self.num_labels)
                    active_labels = torch.where(
                        active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                    )
                    loss = loss_fct(active_logits, active_labels)
                else:
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            if not return_dict:
                output = (logits,) + outputs[1:]
                return ((loss,) + output) if loss is not None else output

            return TokenClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

    def align_predictions(predictions, label_ids):
        preds = np.argmax(predictions, axis=2)
        batch_size, seq_len = preds.shape

        out_label_list = [[] for _ in range(batch_size)]
        preds_list = [[] for _ in range(batch_size)]

        for i in range(batch_size):
            for j in range(seq_len):
                if label_ids[i, j] != torch.nn.CrossEntropyLoss().ignore_index:
                    out_label_list[i].append(id2tag[label_ids[i][j]])
                    preds_list[i].append(id2tag[preds[i][j]])

        return preds_list, out_label_list

    def compute_metrics(p):
        preds_list, out_label_list = align_predictions(p.predictions, p.label_ids)
        return {
            'accuracy': accuracy_score(out_label_list, preds_list),
            'precision': precision_score(out_label_list, preds_list),
            'recall': recall_score(out_label_list, preds_list),
            'f1': f1_score(out_label_list, preds_list),
        }

    training_args = TrainingArguments(
        output_dir='./results/',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01
    )

    model = TSDModel.from_pretrained(model_name_or_path, num_labels=len(unique_tags))

    if do_train:
        trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
        trainer.train()

        if not os.path.exists(model_output_dir):
            os.makedirs(model_output_dir)

        trainer.save_model(model_output_dir)

    if do_eval:
        trainer = Trainer(model=model, args=training_args, eval_dataset=dataset, compute_metrics=compute_metrics)
        metrics = trainer.evaluate()

        if not os.path.exists(result_output_dir):
            os.makedirs(result_output_dir)

        with open(f'{result_output_dir}eval_results.txt', 'w') as writer:
            for key, value in metrics.items():
                writer.write(f'{key} = {value}\n')

    if do_test:
        trainer = Trainer(model=model, args=training_args, compute_metrics=compute_metrics)
        preds, label_ids, metrics = trainer.predict(dataset)
        preds_list, _ = align_predictions(preds, label_ids)

        if not os.path.exists(result_output_dir):
            os.makedirs(result_output_dir)

        with open(f'{result_output_dir}test_results.txt', 'w') as writer:
            for key, value in metrics.items():
                writer.write(f'{key} = {value}\n')

        with open(f'{result_output_dir}test_predictions.csv', 'w') as file:
            writer = csv.writer(file)
            writer.writerow(['spans', 'text'])

            for text in pd.read_csv(file_path)['text']:
                original_tokens = re.findall(r"\w+(?:'\w+)*|[^\w]", text)
                spans = []

                for tokens, preds in zip(texts, preds_list):
                    if tokens == [token for token in original_tokens if not token.isspace()]:
                        break

                char_offset = list_offset = 0
                for i, original_token in enumerate(original_tokens):
                    length = len(original_token)
                    if original_token.isspace() or not original_token.encode('ascii', 'ignore').decode('ascii'):
                        char_offset += length
                        list_offset += 1
                        continue

                    if preds[i - list_offset] == 'B-toxic' or preds[i - list_offset] == 'I-toxic':
                        spans.extend(list(range(char_offset, char_offset + length)))

                    char_offset += length

                writer.writerow([spans, text])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('file_path')
    parser.add_argument('--do_train', action='store_true', default=False)
    parser.add_argument('--do_eval', action='store_true', default=False)
    parser.add_argument('--do_test', action='store_true', default=False)
    parser.add_argument('--model_name_or_path', default='distilbert-base-uncased')
    parser.add_argument('--tokenizer_name_or_path', default='distilbert-base-uncased')
    parser.add_argument('--model_output_dir')
    parser.add_argument('--result_output_dir')
    parser.add_argument('--tfidf', action='store_true', default=False)

    args = parser.parse_args()

    if args.do_train and args.model_output_dir is None:
        raise ValueError('--do_train requires --model_output_dir')
    if (args.do_eval or args.do_test) and args.result_output_dir is None:
        raise ValueError('--do_eval and --do_test require --result_output_dir')

    main(
        file_path=args.file_path,
        do_train=args.do_train,
        do_eval=args.do_eval,
        do_test=args.do_test,
        model_name_or_path=args.model_name_or_path,
        tokenizer_name_or_path=args.tokenizer_name_or_path,
        model_output_dir=args.model_output_dir,
        result_output_dir=args.result_output_dir,
        tfidf=args.tfidf,
    )
