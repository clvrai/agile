import torch
import numpy as np
from torch import nn

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GAN_LSTM(nn.Module):
    """ GAN LSTM in CDQN paper """

    def __init__(self, dim_in=28, dim_hidden=20, mlp_dim_in=1600, dim_out=1, num_layers=1, device=str(DEVICE)):
        super(GAN_LSTM, self).__init__()
        self._dim_in = dim_in
        self._dim_out = dim_out
        self._mlp_dim_in = mlp_dim_in
        self._dim_hidden = dim_hidden
        self._num_layers = num_layers
        self._device = device

        # LSTM(State Representation Module)
        self.lstm = nn.LSTM(input_size=dim_in, hidden_size=dim_hidden, num_layers=num_layers, batch_first=True)

        # Reward Model: Architecture follows the original implementation
        self.dense1 = nn.Linear(mlp_dim_in, 64)
        self.act1 = nn.ELU()
        self.dense2 = nn.Linear(64, 64)
        self.act2 = nn.ELU()
        # this is the slate size to be displayed and can be different from the slate size in a log data
        self.dense3 = nn.Linear(64, dim_out)

    def forward(self, inputs):
        history_seq, slates, clicked_item_position = inputs
        clicked_item_position = clicked_item_position.cpu().detach().numpy()

        # Set initial hidden and cell states
        h0 = torch.zeros(self._num_layers, history_seq.size(0), self._dim_hidden).to(self._device)
        c0 = torch.zeros(self._num_layers, history_seq.size(0), self._dim_hidden).to(self._device)

        # Forward propagate LSTM
        processed_states, _ = self.lstm(history_seq, (h0, c0))  # out: batch_size x seq_length x hidden_size

        # Preprocess the states and the slates for Reward Model
        """ I know it's tricky but we need to compute the likelihood of click for each item in the given slate
            So that I repeated the processed seq of item embeddings of click items from t-m to t-1.
            Then we concatenate the displayed slates at t to the processed states.
        """
        batch_size, history_size, dim_item = processed_states.shape
        batch_size, slate_size, dim_item = slates.shape

        # for each elem in a batch, we will horizontally concat the item embeddings
        processed_states = processed_states.reshape(batch_size, history_size * dim_item)

        # Repeat the concatenated item embeddings for each item in a slate
        # batch_size x slate_size x history_size*dim_item
        processed_states = processed_states.unsqueeze(dim=1).repeat((1, slate_size, 1))

        # Insert the item embedding in a slate to the processed historical item sequence
        _input = torch.cat([processed_states, slates], dim=-1)  # batch_size x slate_size x (history_size+1)*dim_item

        # Reshape the format of the tensor: Vertically rollout the slates
        _input = _input.reshape(batch_size * slate_size, self._mlp_dim_in)

        # MLP to produce the score for items in each slate
        out = self.act1(self.dense1(_input))
        out = self.act2(self.dense2(out))
        out = self.dense3(out)
        out = out.reshape((batch_size, slate_size, self._dim_out))
        # Numerator
        exp_u_disp = torch.exp(out)  # batch_size x slate_size x dim_out
        exp_u_disp = torch.sum(exp_u_disp, dim=-1)  # batch_size x slate_size
        numerator = torch.sum(exp_u_disp, dim=-1)  # (batch_size)

        # Denominator
        _out = torch.sum(out, dim=-1)  # batch_size x slate_size
        # use gather like this; _q_i = q_i.gather(dim=1, index=a_i)  # batch_size x 1
        _clicked_item_position = torch.eye(slate_size)[clicked_item_position]  # batch_size x SLATE_SIZE
        denominator = (_out * _clicked_item_position).sum(dim=-1)  # (batch_size)

        # Compute the loss
        loss_sum = torch.sum(torch.log(numerator + 1) - denominator)
        loss = loss_sum / batch_size

        # Compute Precision1 and 2
        top_2 = torch.topk(_out, k=2, sorted=False)[1].cpu().detach().numpy()  # batch_size x 2
        # prec1_sum = np.sum(np.equal(clicked_item_position[:, None], top_2[:, 0])).astype(np.float)
        prec1_sum = np.sum(np.equal(clicked_item_position, top_2[:, 0])).astype(np.float)
        prec1 = prec1_sum / (batch_size * 1.0)
        # prec2_sum = np.sum(np.equal(clicked_item_position[:, None], top_2[:, 1]))
        prec2_sum = np.sum(np.equal(clicked_item_position, top_2[:, 1])).astype(np.float)
        prec2 = (prec1_sum + prec2_sum) / (batch_size * 1.0)

        # Reshape the output because we just repeated over slate size we can remove this dim when predict
        out = out.reshape(batch_size, SLATE_SIZE, self._dim_out)
        out = out[:, 0, :].squeeze(dim=1)
        return out, {"loss": loss, "prec1": prec1, "prec2": prec2}


def _test_UserModel():
    """ test method """

    history_size = 3
    step_size = 5
    num_user = 3
    SLATE_SIZE = 40
    display_SLATE_SIZE = 5
    dim_item = 10

    history_seq = torch.randn(step_size * num_user, history_size, dim_item)
    slates = torch.randn(step_size * num_user, SLATE_SIZE, dim_item)
    clicked_item_position = torch.tensor(np.random.randint(low=0, high=SLATE_SIZE, size=step_size * num_user))

    model = UserModel(dim_in=dim_item,
                      dim_hidden=dim_item,
                      mlp_dim_in=(history_size + 1) * dim_item,
                      dim_out=display_SLATE_SIZE,
                      device="cpu")
    output, info = model([history_seq, slates, clicked_item_position])
    print(output.shape)
    print("[Eval] Loss: {} Prec@1: {} Prec@2: {}".format(info["loss"].item(), info["prec1"], info["prec2"]))


if __name__ == '__main__':
    _test_UserModel()
