# batch_bilstm_crf

### Model parameters

- vocab_size: the size of vocabulary, because of <PAD> tag, usually we set size + 1
- tagset: tag list, like ['O', 'B-PERSON', 'I-PERSON', 'START', 'END']. This model need start tag and end tag! Attention !
- embedding_dim: the dim of embedding
- hidden_dim: the dim of hidden state
- num_layers: number of recurrent layers
- bidirectional: If True, becomes a BiLSTM.
- dropout: dropout rate, usually we choose 0.1
- start_tag: string of start tag, default: 'START'
- end_tag: string of end tag, default: 'END'
- device: torch.device('cpu') || torch.device('gpu')
- pretrained: torch.nn.Embedding.from_pretrained(pretrained)

### Input Data Structure

- batch_input: tensor of pad_sequence
- batch_input_lens: tensor of each sentence length
- batch_mask: mask of batch input and flatten
- batch_target: list of batch_input tagging

For example:
- batch_input: torch.Tensor([[1, 2, 3, 4], [5, 6, 7, 0], [8, 9, 0, 0]]), 0 means PAD.
- batch_input: torch.Tensor([4, 3, 2])
- batch_mask: [True, True, True, True, True, True, True, False, True, True, False, False]
- batch_target: [torch.Tensor([3, 1, 1, 1]), torch.Tensor([2, 0, 0]), torch.Tensor([3, 5])]

### traning loop snippet

```python
model.init_hidden(batch_size, device)
for batch_info in train_iter:
    batch_input, batch_input_lens, batch_mask, batch_target = batch_info
    loss_train = model.neg_log_likelihood(batch_input, batch_input_lens, batch_mask, batch_target)
    optimizer.zero_grad()
    loss_train.backward()
    optimizer.step()

```
### evaluating snippet

```python
model.init_hidden(batch_size, device)
for batch_info in test_iter:
    batch_input, batch_input_lens, batch_mask, batch_target = batch_info
    batch_pred = model.predict(batch_input, batch_input_lens, batch_mask)
    loss_test = loss_fn(batch_input, batch_input_lens, batch_mask, batch_target)
```



