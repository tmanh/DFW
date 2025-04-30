import os
import logging
import pickle
import numpy as np

if os.path.exists('log-test-all.txt'):
    os.remove('log-test-all.txt')
logging.basicConfig(filename="log-test-all.txt", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_and_log(path):
    with open(path, 'rb') as f:
        result = pickle.load(f)

    errors = []
    for k in result:
        if result[k]['count'] > 0:
            errors.append(result[k]['loss'] / result[k]['count'])

    perrors = []
    for k in result:
        if result[k]['count'] > 0:
            perrors.append(result[k]['loss-p'] / result[k]['count'])

    errors = np.array(errors)
    perrors = np.array(perrors)
    mean_pval = np.mean(perrors)

    logging.info(f'{path}')
    percentiles = [25, 50, 75, 100]
    for p in percentiles:
        threshold = np.percentile(errors, p)
        subset = errors[errors <= threshold]
        fraction = len(subset) / len(errors)
        mean_val = np.mean(subset)
        std_val = np.std(subset)

        logging.info(
            f"Lowest {p}% (threshold: {threshold:.4f}): "
            f"fraction = {fraction:.3f}, mean = {mean_val:.4f}, std = {std_val:.4f}"
        )
    logging.info('\n')


folder = './'

load_and_log(os.path.join(folder, 'bk_checkpoint_all/model.mlp.MLPW-wo-results.pkl'))
load_and_log(os.path.join(folder, 'bk_checkpoint_all/g-model.gru.GRU-io-results.pkl'))
load_and_log(os.path.join(folder, 'bk_checkpoint_all/model.distance.InverseDistance-o-results.pkl'))
load_and_log(os.path.join(folder, 'bk_checkpoint_all/model.kriging.OrdinaryKrigingInterpolation-o-results.pkl'))

load_and_log(os.path.join(folder, 'model.mlp.MLPW-wo-results.pkl'))
load_and_log(os.path.join(folder, 'g-model.gru.GRU-io-results.pkl'))
load_and_log(os.path.join(folder, 'model.distance.InverseDistance-o-results.pkl'))
load_and_log(os.path.join(folder, 'model.kriging.OrdinaryKrigingInterpolation-o-results.pkl'))
