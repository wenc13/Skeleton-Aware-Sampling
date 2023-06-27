import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import h5py


parser = argparse.ArgumentParser()
parser.add_argument('--dump_dir', default='dump', help='dump folder path [default: dump]')
parser.add_argument('--num_out_points', type=int, default=32, help='Number of output points [2, 4, ..., 1024] [default: 32]')

args = parser.parse_args()
args.file_name = 'retrieval_vectors' + '_' + str(args.num_out_points)
args.retrieval_data_path = os.path.join(args.dump_dir, args.file_name + '.h5')


def calc_macro_mean_average_precision(retrieval_vectors, labels):
    batch_size, _ = retrieval_vectors.shape
    sum_avg_per = 0
    for b in range(batch_size):
        sum_avg_per += calc_average_precision(retrieval_vectors, labels, b)
    return sum_avg_per / float(batch_size)


def calc_average_precision(vecs, labels, idx):
    dists = (vecs - vecs[idx]) ** 2
    dists = np.sum(dists, axis=1)
    rets = np.argsort(dists)
    matches = (labels[rets] == labels[idx])
    matches_cum = np.cumsum(matches, dtype='float32')
    precision = matches_cum / range(1, labels.shape[0] + 1)

    relevant_idx = np.where(matches)[0]
    precision_relevant = precision[relevant_idx]
    return np.sum(precision_relevant) / float(np.size(precision_relevant))


def print_precision_recall_to_file(retrieval_vectors, labels):
    precision_file_path = os.path.join(args.dump_dir, 'precision_' + args.file_name + '.txt')
    recall_file_path = os.path.join(args.dump_dir, 'recall_' + args.file_name + '.txt')

    log_fout_percision = open(precision_file_path, 'w')
    log_fout_recall = open(recall_file_path, 'w')

    labels_unique, label_counts = np.unique(labels, return_counts=True)

    batch_size, _ = retrieval_vectors.shape

    percision = np.zeros(100, dtype='float32')
    recall = np.zeros(100, dtype='float32')

    for b in range(batch_size):
        curr_per, curr_rec = precision_recall(retrieval_vectors, labels, label_counts, b)
        percision += curr_per
        recall += curr_rec

    for i in range(100):
        log_string('%f' % (percision[i] / float(batch_size)), log_fout_percision)
        log_string('%f' % (recall[i] / float(batch_size)), log_fout_recall)

    log_fout_percision.close()
    log_fout_recall.close()

    return precision_file_path, recall_file_path


# def print_precision_recall_curve(retrieval_vectors, labels, log_fout_percision, log_fout_recall):
#     labels_unique, label_counts = np.unique(labels, return_counts=True)
#
#     B, _ = retrieval_vectors.shape
#
#     percision = np.zeros(100, dtype='float32')
#     recall = np.zeros(100, dtype='float32')
#
#     for b in range(B):
#         curr_per, curr_rec = precision_recall(retrieval_vectors, labels, label_counts, b)
#         percision += curr_per
#         recall += curr_rec
#
#     for i in range(100):
#         log_string('%f' % (percision[i] / float(B)), log_fout_percision)
#         log_string('%f' % (recall[i] / float(B)), log_fout_recall)


def precision_recall(vecs, labels, labels_counts, idx):
    dists = (vecs - vecs[idx]) ** 2
    dists = np.sum(dists, axis=1)
    rets = np.argsort(dists)
    matches = (labels[rets] == labels[idx])
    matches_cum = np.cumsum(matches, dtype='float32')
    total_class = labels_counts[labels[idx]]

    precision = matches_cum / range(1, labels.shape[0] + 1)
    recall = matches_cum / total_class

    relevant_idx = np.where(matches)[0]
    precision_relevant = precision[relevant_idx]
    recall_relevant = recall[relevant_idx]

    precision_padded = np.pad(precision_relevant, (0, 100 - np.size(precision_relevant)), 'constant',
                              constant_values=(precision_relevant[-1]))
    recall_padded = np.pad(recall_relevant, (0, 100 - np.size(recall_relevant)), 'constant',
                           constant_values=(recall_relevant[-1]))

    return precision_padded, recall_padded


def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return data, label


def log_string(out_str, log_fout):
    log_fout.write(out_str + '\n')
    log_fout.flush()
    print(out_str)


def plot_precision_recall(recall, precision):
    plt.plot(recall, precision, c='g', linewidth=3, markersize=8)

    axis_val = np.arange(0, 5 + 1, 1) / 5.
    axis_str = [str(v) for v in axis_val]

    plt.xticks(axis_val, axis_str)
    plt.yticks(axis_val, axis_str)

    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)

    plt.grid(True)
    plt.tight_layout()


if __name__ == '__main__':
    # load retrieval data
    retrieval_vectors, labels = load_h5(args.retrieval_data_path)

    # calculate mean average precision
    res_macro = calc_macro_mean_average_precision(retrieval_vectors, labels)

    # print the result to log file
    log_fout_retrieval = open(os.path.join(args.dump_dir, 'log_retrieval_' + args.file_name + '.txt'), 'w')
    log_string('macro_mean_average_precision result', log_fout_retrieval)
    log_string('-----------------------------------', log_fout_retrieval)
    log_string('mAP: %f' % res_macro, log_fout_retrieval)
    log_fout_retrieval.close()

    # compute precision recall data
    precision_file_path, recall_file_path = print_precision_recall_to_file(retrieval_vectors, labels)

    # plot precision recall curve
    precision = np.genfromtxt(precision_file_path, delimiter='\n')
    recall = np.genfromtxt(recall_file_path, delimiter='\n')

    fig = plt.figure(figsize=(8, 4.5))
    plot_precision_recall(recall, precision)
    plt.legend(['S-NET - %d points' % args.num_out_points], loc='lower left', fontsize=12)
    fig.savefig(os.path.join(args.dump_dir, 'precision_recall_curve.png'))
    plt.close()
