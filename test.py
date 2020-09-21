import argparse
import torch
from tqdm import tqdm
import model.metric as module_metric
import model.loss as module_loss
from parse_config import ConfigParser
from utils.util import create_model, create_dataloader


def main(config):
    logger = config.get_logger('test')
    model = create_model(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.initialize(config, device)
    
    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    model = model.to(device)
    model.eval()


    metric_fns = [getattr(module_metric, met) for met in config['metrics']]
    loss_fn = getattr(module_loss, config['loss'])
    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    data_loader = create_dataloader(config)
    with torch.no_grad():
        for i, data in enumerate(tqdm(data_loader)):
            target, U_v, U_a, U_t, U_p, M_v, M_a, M_t, target_loc, umask, seg_len, n_c = [d.to(device) for d in data]
            seq_lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))]
            output = model(U_v, U_a, U_t, U_p, M_v, M_a, M_t, seq_lengths, target_loc, seg_len, n_c)
            target = target.squeeze(1)
            loss = loss_fn(output, target)
            batch_size = U_v.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(output, target) * batch_size

    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    logger.info(log)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
