import torch
from base import BaseTrainer
import datetime
from utils import inf_loop, MetricTracker


class MEmoRTrainer(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, config, data_loader,
                 valid_data_loader=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, config)
        self.config = config
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, self.optimizer)
        self.log_step = 200        
        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, data in enumerate(self.data_loader):          
            target, U_v, U_a, U_t, U_p, M_v, M_a, M_t, target_loc, umask, seg_len, n_c = [d.to(self.device) for d in data]
            
            self.optimizer.zero_grad()
            seq_lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))]
            
            output = self.model(U_v, U_a, U_t, U_p, M_v, M_a, M_t, seq_lengths, target_loc, seg_len, n_c)
            assert output.shape[0] == target.shape[0]
            target = target.squeeze(1)
            loss = self.criterion(output, target)
            loss.backward()

            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f} Time:{}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item(),
                    datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        outputs, targets= [], []
        with torch.no_grad():
            for batch_idx, data in enumerate(self.valid_data_loader):
                target, U_v, U_a, U_t, U_p, M_v, M_a, M_t, target_loc, umask, seg_len, n_c = [d.to(self.device) for d in data]
                seq_lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))]
            
                output = self.model(U_v, U_a, U_t, U_p, M_v, M_a, M_t, seq_lengths, target_loc, seg_len, n_c)
                target = target.squeeze(1)
                loss = self.criterion(output, target)
                
                outputs.append(output.detach())
                targets.append(target.detach())
  
                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                            
            outputs = torch.cat(outputs, dim=0)
            targets = torch.cat(targets, dim=0)
            for met in self.metric_ftns:
                self.valid_metrics.update(met.__name__, met(outputs, targets))
        
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)