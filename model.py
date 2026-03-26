import torch
import torch.nn as nn

class Seq2SeqTransformer(nn.Module):
    def __init__(self, vocab_size=51, embed_size=128, num_heads=8, num_layers=6, max_len=1000):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.register_buffer('pos_embed', torch.zeros(max_len, embed_size))
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_size, nhead=num_heads, batch_first=True),
            num_layers=num_layers
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=embed_size, nhead=num_heads, batch_first=True),
            num_layers=num_layers
        )
        self.fc = nn.Linear(embed_size, vocab_size)

    def forward(self, src, tgt):
        src_len = src.size(1)
        tgt_len = tgt.size(1)
        batch_size = src.size(0)
        src_pos = self.pos_embed[:src_len].unsqueeze(0).expand(batch_size, -1, -1)
        tgt_pos = self.pos_embed[:tgt_len].unsqueeze(0).expand(batch_size, -1, -1)
        src_embed = self.embed(src) + src_pos
        tgt_embed = self.embed(tgt) + tgt_pos
        src_mask = None
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).to(src.device)
        memory = self.encoder(src_embed)
        output = self.decoder(tgt_embed, memory, tgt_mask=tgt_mask)
        return self.fc(output)

    def generate(self, src, max_len=7, start_token=50):
        # For inference, generate the next 7 tokens autoregressively
        self.eval()
        with torch.no_grad():
            src_len = src.size(1)
            batch_size = src.size(0)
            src_pos = self.pos_embed[:src_len].unsqueeze(0).expand(batch_size, -1, -1)
            src_embed = self.embed(src) + src_pos
            memory = self.encoder(src_embed)
            tgt = torch.full((batch_size, 1), start_token, dtype=torch.long, device=src.device)
            for _ in range(max_len):
                tgt_len = tgt.size(1)
                tgt_pos = self.pos_embed[:tgt_len].unsqueeze(0).expand(batch_size, -1, -1)
                tgt_embed = self.embed(tgt) + tgt_pos
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).to(src.device)
                output = self.decoder(tgt_embed, memory, tgt_mask=tgt_mask)
                next_token = self.fc(output[:, -1, :]).argmax(dim=-1, keepdim=True)
                tgt = torch.cat([tgt, next_token], dim=1)
            return tgt[:, 1:]  # the 7 tokens