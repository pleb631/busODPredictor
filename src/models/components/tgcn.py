import torch
from torch import nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_dim).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim)
        ).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[:, : x.size(1)].expand_as(x)


class PatchEmbedding_time(nn.Module):
    def __init__(self, d_model=512, patch_len=1, stride=1, his=7):
        super(PatchEmbedding_time, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.his = his
        self.daytime_embedding = nn.Embedding(24 + 1, d_model // 3)
        weekday_size = 7 + 1
        self.weekday_embedding = nn.Embedding(weekday_size, d_model // 3)

        self.month_embedding = nn.Embedding(12 + 1, d_model -2*(d_model// 3))

    def forward(self, x, his=True):
        # do patching
        bs, dim, ts = x.size()

        if his:
            if self.his == ts:
                x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
            else:
                gap = self.his // ts
                x = x.unfold(
                    dimension=-1, size=self.patch_len // gap, step=self.stride // gap
                )
            num_patch = x.shape[-2]
            x = x.reshape(bs, dim, num_patch, -1)

            # 分块提取第一个元素，目的是减少数据冗余
            x_tdh = x[:, 0, :, 0]
            x_dwh = x[:, 1, :, 0]
            x_m = x[:, 2, :, 0]

            x_tdh = self.daytime_embedding(x_tdh)
            x_dwh = self.weekday_embedding(x_dwh)
            x_m = self.month_embedding(x_m)

        else:
            x = x.permute(0, 2, 1)
            x_tdh = x[..., 0]
            x_dwh = x[..., 1]
            x_m = x[..., 2]
            x_tdh = self.daytime_embedding(x_tdh)
            x_dwh = self.weekday_embedding(x_dwh)
            x_m = self.month_embedding(x_m)

        x_th = torch.cat([x_tdh, x_dwh,x_m], dim=-1)

        return x_th


class PatchEmbedding_flow(nn.Module):
    def __init__(self, d_model=256, dim_in=7, patch_len=1, stride=1, his=7):
        super(PatchEmbedding_flow, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.his = his
        self.dim = patch_len * dim_in
        self.value_embedding = nn.Linear(patch_len * dim_in, d_model, bias=False)
        self.position_encoding = PositionalEncoding(d_model)

    def forward(self, x):
        # do patching
        b, c, n = x.shape
        if self.his == n:
            x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        else:
            gap = self.his // n
            x = x.unfold(
                dimension=-1, size=self.patch_len // gap, step=self.stride // gap
            )
            x = F.pad(x, (0, (self.patch_len - self.patch_len // gap)))

        x = x.permute(0, 2, 1, 3).reshape(b, -1, self.dim)
        x = self.value_embedding(x)
        x = x + self.position_encoding(x)
        return x


class TGCN(nn.Module):

    def __init__(
        self,
        dim_in=13,
        output_dim=7,
        input_window=14,
        embed_dim=128,
    ) -> None:

        super().__init__()

        self.dim_in = dim_in
        self.output_dim = output_dim
        self.embed_dim = embed_dim

        self.patch_embedding_time = PatchEmbedding_time(
            self.embed_dim, patch_len=1, stride=1, his=input_window
        )

        self.patch_embedding_flow = PatchEmbedding_flow(
            self.embed_dim, patch_len=1, stride=1, his=input_window, dim_in=dim_in
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=4,
            dim_feedforward=self.embed_dim * 2,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.embed_dim,
            nhead=4,
            dim_feedforward=self.embed_dim * 2,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)

        self.proj = nn.Linear(self.embed_dim * 2, self.embed_dim)

        self.predict = nn.Linear(self.embed_dim, 1)

    def forward(self, data: torch.Tensor, lbls) -> torch.Tensor:

        TCH = data[:, self.dim_in :].long()
        TCP = lbls[:, self.dim_in :].long()  # 预测标签，训练代码中只使用日期和时间信息
        feas_all_his = self.patch_embedding_time(TCH)
        feas_all_pre = self.patch_embedding_time(TCP, his=False)

        x_in = data[:, : self.dim_in]
        means = x_in[:,0:1,:].mean(-1, keepdim=True).detach()
        x_in[:,0:1,:] = x_in[:,0:1,:] - means
        stdev = torch.sqrt(
            torch.var(x_in[:,0:1,:], dim=-1, keepdim=True, unbiased=False) + 1e-5
        ).detach()
        x_in[:,0:1,:] /= stdev

        enc = self.patch_embedding_flow(x_in)

        feat = self.proj(torch.cat([enc, feas_all_his], dim=-1))
        feat = self.encoder(feat)
        out = self.decoder(feas_all_pre, feat)

        out = self.predict(out)
        out = out * stdev + means

        return out.squeeze(-1)


if __name__ == "__main__":
    model = TGCN()
    data = torch.randint(1, 7, (2, 15, 14), dtype=torch.float32)
    lbls = torch.randint(1, 7, (2, 15, 7), dtype=torch.float32)
    OUT = model(data, lbls)
    print(OUT.shape)
