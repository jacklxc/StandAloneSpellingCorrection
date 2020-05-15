import paddle
import paddle.fluid as fluid
from LSTM import SimpleLSTMRNN
from datasetUtil import readVocab, loadNoisyInputLabel, all_sentence2examples, \
    charReplacement, wordReplacement, excludeSentenceWithUnknownWords

import json
import numpy as np
import random
from copy import copy

import argparse

class ModelHyperParams(object):
    """
    ModelHyperParams
    """
    hidden_size = 650
    vocab_size = 50003
    batch_size = 32
    learning_rate = 0.1
    unk_idx = vocab_size - 2 
    oov_idx = vocab_size - 1
    limit = 3e5
    max_sentence_len = 200
    warmup_steps = 100
    n_split = 5
    dynamic = False
    mode = "natural"
    train_replacement = "train_replacement.json"
    log_file = "robust_performance.log"

def hasnum(w):
    for c_i in w:
        if c_i.isdigit():
            return True
    return False

def word2feature(w):
    bin_initial = [0]*len(alph)
    bin_middle = [0]*len(alph)
    bin_end = [0]*len(alph)
    if w == '<eos>':
        bin_initial[-1] += 1
        bin_middle[-1] += 1
        bin_end[-1] += 1
    elif w == '<unk>':
        bin_initial[-2] += 1
        bin_middle[-2] += 1
        bin_end[-2] += 1
    elif hasnum(w):
        bin_initial[-3] += 1
        bin_middle[-3] += 1
        bin_end[-3] += 1
    else:
        for i in range(len(w)):
            try:
                if i==0:
                    bin_initial[alph.get(w[i],-2)] += 1
                elif i==len(w)-1:
                    bin_end[alph.get(w[i],-2)] += 1
                else:
                    bin_middle[alph.get(w[i],-2)] += 1
            except ValueError:
                print(w)
                raise
    bin_all = bin_initial + bin_middle + bin_end
    return np.array(bin_all)

def dynamicReaderCreator(input_path, word_replacement, mode = "natural"):
    dataset = []
    with open(input_path,"r") as f:
        for line in f:
            tokens_1B = line.strip().split()
            dataset.append(tokens_1B)

    input_size = len(dataset)
    block_size = input_size // ModelHyperParams.n_split

    def reader():
        for j in range(ModelHyperParams.n_split):
            if mode == "mix":
                replaced_dataset, gold_labels = wordReplacement(word_replacement, dataset[j * block_size : (j+1) * block_size], word_vocab = None, sigma=0.2)
                char_replaced_dataset, char_gold_labels = charReplacement(dataset[j * block_size : (j+1) * block_size], word_vocab = None, sigma=0.2)
                replaced_dataset += char_replaced_dataset
                gold_labels += char_gold_labels

            elif mode == "char":
                replaced_dataset, gold_labels = charReplacement(dataset[j * block_size : (j+1) * block_size], word_vocab = None, sigma=0.2)
            
            else: # natural
                replaced_dataset, gold_labels = wordReplacement(word_replacement, dataset[j * block_size : (j+1) * block_size], word_vocab = None, sigma=0.2)

            indices = [i for i in range(len(replaced_dataset))]
            random.shuffle(indices)
            for i in indices:
                input_tokens = replaced_dataset[i]
                correct_tokens = copy(input_tokens)
                for index, correct_word in gold_labels[i].items():
                    correct_tokens[index] = correct_word
                yield (input_tokens, correct_tokens, gold_labels[i])

    return reader

def staticReaderCreator(input_path, label_path):

    def reader():
        with open(input_path, "r") as f:
            with open(label_path, "r") as lf:
                inputs = []
                labels = []
                count = 0
                for line, label_line in zip(f, lf):
                    count += 1
                    tokens = line.strip().split()
                    label = eval(label_line.strip())
                    label = {int(k):v for k, v in label.items()}
                    inputs.append(tokens)
                    labels.append(label)

                    if count >= ModelHyperParams.limit:
                        indices = [i for i in range(len(inputs))]
                        random.shuffle(indices)

                        for i in indices:
                            input_tokens = inputs[i]
                            correct_tokens = copy(input_tokens)
                            for index, correct_word in labels[i].items():
                                correct_tokens[index] = correct_word
                            yield (input_tokens, correct_tokens, labels[i])

                        inputs = []
                        labels = []
                        count = 0

                indices = [i for i in range(len(inputs))]
                #random.shuffle(indices)
                for i in indices:
                    input_tokens = inputs[i]
                    correct_tokens = copy(input_tokens)
                    for index, correct_word in labels[i].items():
                        correct_tokens[index] = correct_word
                    yield (input_tokens, correct_tokens, labels[i])
    return reader

def prepare_batch_input(batch):
    X = np.zeros((ModelHyperParams.batch_size, ModelHyperParams.max_sentence_len, len(alph)*3), dtype=np.float32)
    Y = np.zeros((ModelHyperParams.batch_size, ModelHyperParams.max_sentence_len, 1), dtype=np.int64) + ModelHyperParams.oov_idx

    for si, example in enumerate(batch):
        input_tokens, correct_tokens, _ = example
        sentence_vector = []
        for token in input_tokens:
            vector = word2feature(token)
            sentence_vector.append(vector)
        sentence_vector = np.array(sentence_vector)
        X[si,:sentence_vector.shape[0],:] = sentence_vector
        for wi, token in enumerate(correct_tokens):
            Y[si, wi, 0]  = word_vocab.get(token, ModelHyperParams.oov_idx)
    return X, Y

def create_data(X, Y):
    X_paddle = fluid.dygraph.to_variable(X)
    Y_paddle = fluid.dygraph.to_variable(Y)
    return X_paddle, Y_paddle

class RobustCorrector(fluid.Layer):
    """
    model
    """
    def __init__(self, input_size, hidden_size, vocab_size, batch_size, num_steps, num_layers=1, dropout=0.01):
        super(RobustCorrector, self).__init__('spelling_corrector')
        self.dropout = dropout
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.num_steps = num_steps
        self.input_size = input_size

        self.projection_weight = self.create_parameter(
            attr=fluid.ParamAttr(),
            dtype="float32",
            shape=[self.input_size, self.hidden_size],
            default_initializer=fluid.initializer.XavierInitializer())
        self.projection_bias = self.create_parameter(
            attr=fluid.ParamAttr(),
            dtype="float32",
            shape=[self.hidden_size],
            default_initializer=fluid.initializer.XavierInitializer())

        self.simple_lstm_rnn = SimpleLSTMRNN(
            hidden_size,
            num_steps,
            num_layers=num_layers,
            dropout=None)

        self.softmax_weight = self.create_parameter(
            attr=fluid.ParamAttr(),
            dtype="float32",
            shape=[self.hidden_size, self.vocab_size],
            default_initializer=fluid.initializer.XavierInitializer())
        self.softmax_bias = self.create_parameter(
            attr=fluid.ParamAttr(),
            dtype="float32",
            shape=[self.vocab_size],
            default_initializer=fluid.initializer.XavierInitializer())

    def forward(self, input, label):
        init_h = fluid.layers.fill_constant([self.num_layers, self.batch_size, self.hidden_size], 'float32', 0.0)

        init_c = fluid.layers.fill_constant([self.num_layers, self.batch_size, self.hidden_size], 'float32', 0.0)

        projection = fluid.layers.matmul(input, self.projection_weight)
        projection = fluid.layers.elementwise_add(projection, self.projection_bias)

        projection = fluid.layers.reshape(
            projection, shape=[-1, self.num_steps, self.hidden_size])

        rnn_out, _, _ = self.simple_lstm_rnn(projection, init_h, init_c)
        rnn_out = fluid.layers.dropout(
                rnn_out,
                dropout_prob=self.dropout,
                dropout_implementation='upscale_in_train')
        prediction = fluid.layers.matmul(rnn_out, self.softmax_weight)
        prediction = fluid.layers.elementwise_add(prediction, self.softmax_bias)
        prediction = fluid.layers.softmax(prediction, axis=-1)
        label_out = fluid.layers.one_hot(input=label, depth=self.vocab_size)
        #label_out = fluid.layers.label_smooth(
        #                label=fluid.layers.one_hot(
        #                    input=label, depth=self.vocab_size),
        #                epsilon=0.1)
        loss = fluid.layers.cross_entropy(
            prediction, label=label_out, soft_label=True)
        loss = fluid.layers.reshape(loss, shape=[-1, self.num_steps])
        loss = fluid.layers.reduce_mean(loss, dim=[0])
        loss = fluid.layers.reduce_sum(loss)
        return loss, prediction

def scores(TP, FP, FN, TN, beta = 0.5):
    precision = TP / (TP+FP)
    recall = TP / (TP+FN)
    accuracy = (TP + TN)/(TP+FP+FN+TN)
    F_score = (1+beta**2) * (precision * recall) / (beta**2 * precision + recall)
    return precision, recall, accuracy, F_score

def computePerformance(corrected_sentences, original_sentences, ground_truth_labels):
    TP = 0
    FP = 0
    FN = 0
    TN = 0

    real_TP = 0
    real_FP = 0
    real_FN = 0
    real_TN = 0

    non_TP = 0
    non_FP = 0
    non_FN = 0
    non_TN = 0

    for tokens, prediction, labels in zip(original_sentences, corrected_sentences, ground_truth_labels):
        correct_tokens = copy(tokens)
        for i, gold in labels.items():
            correct_tokens[i] = gold
        for token, gold, pred in zip(tokens, correct_tokens, prediction):
            if token!=pred:
                if pred==gold:
                    TP += 1
                    if token in word_vocab:
                        real_TP += 1
                    else:
                        non_TP += 1
                else:
                    FP += 1
                    if token in word_vocab:
                        real_FP += 1
                    else:
                        non_FP += 1
            else:
                if pred==gold:
                    TN += 1
                    if token in word_vocab:
                        real_TN += 1
                    else:
                        non_TN += 1
                else:
                    FN += 1
                    if token in word_vocab:
                        real_FN += 1
                    else:
                        non_FN += 1
    precision, recall, accuracy, F_score = scores(TP, FP, FN, TN, beta = 0.5)

    print("Accuracy:", accuracy)
    print("Precision:",precision)
    print("Recall:",recall)
    print("F 0.5:",  F_score)

    real_precision, real_recall, real_accuracy, real_F_score = scores(real_TP, real_FP, real_FN, real_TN, beta = 0.5)

    print("Real-word Accuracy:", real_accuracy)
    print("Real-word Precision:",real_precision)
    print("Real-word Recall:",real_recall)
    print("Real-word F 0.5:",  real_F_score)

    non_precision, non_recall, non_accuracy, non_F_score = scores(non_TP, non_FP, non_FN, non_TN, beta = 0.5)

    print("Non-word Accuracy:", non_accuracy)
    print("Non-word Precision:",non_precision)
    print("Non-word Recall:",non_recall)
    print("Non-word F 0.5:",  non_F_score)

def evaluate(reader, model, word_wise = True):
    correct = 0
    total = 0
    avg_loss_set = []
    all_predicted_sentences = []
    all_original_sentences = []
    all_ground_truth = []
    for batch in reader():
        batch_X_np, batch_Y_np = prepare_batch_input(batch)
        word_inputs, label_var = create_data(batch_X_np, batch_Y_np)
        dy_avg_cost, dy_predict = model(word_inputs, label_var)

        avg_loss_set.append(float(dy_avg_cost.numpy()))

        batch_prediction = np.argmax(dy_predict.numpy(), axis=-1)
        label_np = label_var.numpy().reshape(-1,ModelHyperParams.max_sentence_len)

        correct += np.sum(batch_prediction == label_np)
        total += label_np.size

        for batch_i, example in enumerate(batch):
            input_tokens, _, ground_truth = example
            predicted_tokens = [reverse_word_vocab[index] for index in batch_prediction[batch_i,:len(input_tokens)]]
            all_predicted_sentences.append(predicted_tokens)
            all_original_sentences.append(input_tokens)
            all_ground_truth.append(ground_truth)
    computePerformance(all_predicted_sentences, all_original_sentences, all_ground_truth)
    return correct / total, np.mean(avg_loss_set)

def train(dynamic = ModelHyperParams.dynamic, model_path = None):

    with fluid.dygraph.guard():
        spelling_corrector = RobustCorrector(len(alph)*3, ModelHyperParams.hidden_size, 
            ModelHyperParams.vocab_size, ModelHyperParams.batch_size, 
            ModelHyperParams.max_sentence_len, num_layers=1, dropout=0.01)
        
        if model_path is not None:
            try:
                model_dict, _ = fluid.dygraph.load_dygraph(model_path)
                spelling_corrector.load_dict(model_dict)
                print("Model loaded.")
            except:
                print("Model not found, training from scratch.")
        
        lr = fluid.layers.learning_rate_scheduler.noam_decay(
            1/(ModelHyperParams.warmup_steps *(ModelHyperParams.learning_rate ** 2)),
            ModelHyperParams.warmup_steps)
        optimizer = fluid.optimizer.AdamOptimizer(lr)

        if dynamic:
            train_reader = dynamicReaderCreator(train_input_path, trainWordReplacement, mode = ModelHyperParams.mode)
        else:
            train_reader = staticReaderCreator(train_input_path, train_label_path)
        train_reader = paddle.batch(train_reader, batch_size=ModelHyperParams.batch_size)

        test_reader = staticReaderCreator(inference_input_path, inference_label_path)
        test_reader = paddle.batch(test_reader, batch_size=ModelHyperParams.batch_size)

        best_acc = 0
        for i in range(200):
            dy_step = 0
            avg_costs = []
            for batch in train_reader():
                batch_X_np, batch_Y_np = prepare_batch_input(batch)
                word_inputs, label_var = create_data(batch_X_np, batch_Y_np)
                dy_avg_cost, dy_predict = spelling_corrector(word_inputs, label_var)
                dy_avg_cost.backward()
                optimizer.minimize(dy_avg_cost)
                spelling_corrector.clear_gradients()
                dy_step = dy_step + 1
                dy_avg_cost_float = float(dy_avg_cost.numpy())
                avg_costs.append(dy_avg_cost_float)
                if dy_step % 100 == 0:
                    print("pass num : {}, batch_id: {}, dy_graph avg loss: {}".
                        format(i, dy_step, dy_avg_cost_float))
                if dy_step % 1000 == 0:
                    print("Periodically save the model.")
                    fluid.dygraph.save_dygraph(spelling_corrector.state_dict(), "robust_period")
            print("Pass: {} finished. Avg training cost: {}".format(i, np.mean(avg_costs)))
            
            spelling_corrector.eval()
            test_acc, test_loss = evaluate(test_reader, spelling_corrector,False)
            spelling_corrector.train()
            print("Loss at epoch {} , Test avg_loss is: {}, acc is: {}".format(
                i, test_loss, test_acc))
            if best_acc <= test_acc:
                fluid.dygraph.save_dygraph(spelling_corrector.state_dict(), "robust_best")
                print("New best accuracy. New model saved.")
                best_acc = test_acc
            else:
                print("Accuracy not improved. Skip saving model.")

def train_parallel(dynamic = ModelHyperParams.dynamic, model_path = None):

    place = fluid.CUDAPlace(fluid.dygraph.parallel.Env().dev_id)
    with fluid.dygraph.guard(place):
        strategy = fluid.dygraph.parallel.prepare_context()

        spelling_corrector = RobustCorrector(len(alph)*3, ModelHyperParams.hidden_size, 
            ModelHyperParams.vocab_size, ModelHyperParams.batch_size, 
            ModelHyperParams.max_sentence_len, num_layers=1, dropout=0.01)

        if model_path is not None:
            try:
                model_dict, _ = fluid.dygraph.load_dygraph(model_path)
                spelling_corrector.load_dict(model_dict)
                print("Model loaded.")
            except:
                print("Model not found, training from scratch.")
        
        spelling_corrector = fluid.dygraph.parallel.DataParallel(spelling_corrector, strategy)

        lr = fluid.layers.learning_rate_scheduler.noam_decay(
            1/(ModelHyperParams.warmup_steps *(ModelHyperParams.learning_rate ** 2)),
            ModelHyperParams.warmup_steps)
        optimizer = fluid.optimizer.AdamOptimizer(lr)

        if dynamic:
            train_reader = dynamicReaderCreator(train_input_path, trainWordReplacement, mode = ModelHyperParams.mode)
        else:
            train_reader = staticReaderCreator(train_input_path, train_label_path)
        train_reader = paddle.batch(train_reader, batch_size=ModelHyperParams.batch_size)
        train_reader = fluid.contrib.reader.distributed_batch_reader(train_reader)

        test_reader = staticReaderCreator(inference_input_path, inference_label_path)
        test_reader = paddle.batch(test_reader, batch_size=ModelHyperParams.batch_size)
        
        best_acc = 0
        for i in range(200):
            dy_step = 0
            avg_costs = []
            for batch in train_reader():
                batch_X_np, batch_Y_np = prepare_batch_input(batch)
                word_inputs, label_var = create_data(batch_X_np, batch_Y_np)
                dy_avg_cost, dy_predict = spelling_corrector(word_inputs, label_var)
                dy_avg_cost = spelling_corrector.scale_loss(dy_avg_cost)
                dy_avg_cost.backward()
                optimizer.minimize(dy_avg_cost)
                spelling_corrector.clear_gradients()
                dy_step = dy_step + 1
                dy_avg_cost_float = float(dy_avg_cost.numpy())
                avg_costs.append(dy_avg_cost_float)
                if dy_step % 100 == 0:
                    print("pass num : {}, batch_id: {}, dy_graph avg loss: {}".
                        format(i, dy_step, dy_avg_cost_float))
                if dy_step % 1000 == 0:
                    print("Periodically save the model.")
                    fluid.dygraph.save_dygraph(spelling_corrector.state_dict(), 
                        "robust_"+str(ModelHyperParams.learning_rate)+"_"+str(ModelHyperParams.hidden_size)+"_period")

                    spelling_corrector.eval()
                    test_acc, test_loss = evaluate(test_reader, spelling_corrector,False)
                    spelling_corrector.train()
                    print("Loss at step {} , Test avg_loss is: {}, acc is: {}".format(
                        dy_step, test_loss, test_acc))
                    if best_acc <= test_acc:
                        fluid.dygraph.save_dygraph(spelling_corrector.state_dict(), 
                            "robust_"+str(ModelHyperParams.learning_rate)+"_"+str(ModelHyperParams.hidden_size)+"_best")
                        print("New best accuracy. New model saved.")
                        best_acc = test_acc
                        with open(ModelHyperParams.log_file, "a") as f:
                            log = str(ModelHyperParams.learning_rate)+"\t"+str(ModelHyperParams.hidden_size)+"\t"+str(i)+"\t"+str(best_acc)+"\n"
                            f.write(log)
                    else:
                        print("Accuracy not improved. Skip saving model.")


            print("Pass: {} finished. Avg training cost: {}".format(i, np.mean(avg_costs)))
            
            spelling_corrector.eval()
            test_acc, test_loss = evaluate(test_reader, spelling_corrector,False)
            spelling_corrector.train()
            print("Loss at epoch {} , Test avg_loss is: {}, acc is: {}".format(
                i, test_loss, test_acc))
            if best_acc <= test_acc:
                fluid.dygraph.save_dygraph(spelling_corrector.state_dict(), 
                    "robust_"+str(ModelHyperParams.learning_rate)+"_"+str(ModelHyperParams.hidden_size)+"_best")
                print("New best accuracy. New model saved.")
                best_acc = test_acc
                with open(ModelHyperParams.log_file, "a") as f:
                    log = str(ModelHyperParams.learning_rate)+"\t"+str(ModelHyperParams.hidden_size)+"\t"+str(i)+"\t"+str(best_acc)+"\n"
                    f.write(log)
            else:
                print("Accuracy not improved. Skip saving model.")

def test(model_path):
    with fluid.dygraph.guard():
        spelling_corrector = RobustCorrector(len(alph)*3, ModelHyperParams.hidden_size, 
            ModelHyperParams.vocab_size, ModelHyperParams.batch_size, 
            ModelHyperParams.max_sentence_len, num_layers=1, dropout=0.01)
        
        model_dict, _ = fluid.dygraph.load_dygraph(model_path)
        spelling_corrector.load_dict(model_dict)
        print("Model loaded.")

        test_reader = staticReaderCreator(inference_input_path, inference_label_path)
        test_reader = paddle.batch(test_reader, batch_size=ModelHyperParams.batch_size)
        spelling_corrector.eval()
        test_acc, test_loss = evaluate(test_reader, spelling_corrector, False)
        print(test_acc, test_loss)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", help="Learning rate")
    parser.add_argument('--hidden_size')
    parser.add_argument("--mode")
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()

    if args.lr:
        ModelHyperParams.learning_rate = float(args.lr)
    if args.hidden_size:
        ModelHyperParams.hidden_size = int(args.hidden_size)
    if args.mode:
        ModelHyperParams.mode = args.mode

    alph = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,:;'*!?`$%&(){}[]-/\@_#"
    alph = {a: i for i,a in enumerate(alph)} 
    prefix = "data/"
    train_input_path = prefix+"train.txt"
    train_label_path = prefix+"train_label.jsonl"
    inference_input_path = prefix+"test.txt"
    inference_label_path = prefix+"test_label.jsonl"
    char_path = prefix+"charVocab.txt"
    word_path = prefix+"wordVocab.txt"
    char_vocab, word_vocab = readVocab(char_path, word_path)
    reverse_word_vocab = {index: word for word, index in word_vocab.items()}

    with open(ModelHyperParams.train_replacement) as f:
        trainWordReplacement = json.load(f)

    if args.test:
        test("robust_0.001_650_best")
    else:
        train_parallel(dynamic = ModelHyperParams.dynamic, 
            model_path="robust_"+str(ModelHyperParams.learning_rate)+"_"+str(ModelHyperParams.hidden_size)+"_best")
        
