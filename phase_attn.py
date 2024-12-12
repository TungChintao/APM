import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttentionLayer(nn.Module):
    """Self-attention layer for spatial attention, optimized for lower memory usage."""

    def __init__(self, channels, reduction_ratio=16):
        super(SelfAttentionLayer, self).__init__()
        self.query_conv = nn.Conv2d(channels, channels // reduction_ratio, 1)
        self.key_conv = nn.Conv2d(channels, channels // reduction_ratio, 1)
        self.value_conv = nn.Conv2d(channels, channels, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        # Reducing spatial dimensions for memory efficiency
        downsampled_size = (height // 4, width // 4)
        x_downsampled = F.adaptive_avg_pool2d(x, downsampled_size)

        # Generate query, key, value
        query = self.query_conv(x_downsampled).view(batch_size, -1, downsampled_size[0] * downsampled_size[1])
        key = self.key_conv(x_downsampled).view(batch_size, -1, downsampled_size[0] * downsampled_size[1])
        value = self.value_conv(x_downsampled).view(batch_size, -1, downsampled_size[0] * downsampled_size[1])

        # Compute attention
        attention_scores = torch.bmm(query.permute(0, 2, 1), key)  # Transpose for matrix multiplication
        attention = self.softmax(attention_scores)

        # Apply attention to the value
        out = torch.bmm(value, attention)
        out = out.view(batch_size, channels, downsampled_size[0], downsampled_size[1])

        # Upsample to original size
        out = F.interpolate(out, size=(height, width), mode='bilinear', align_corners=False)

        return out


class PhaseAttention(nn.Module):
    """Channel Attention with Phase Information"""

    def __init__(self, channel, reduction=8):
        super().__init__()
        self.channel = channel
        # self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # Add an average pooling layer
        self.phase_attn = nn.Sequential(
            nn.Linear(channel, reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduction, channel, bias=False),
            nn.Sigmoid()
        )
        # self.amplitude_attn = nn.Sequential(
        #    nn.Linear(channel, reduction, bias=False),
        #    nn.ReLU(inplace=True),
        #    nn.Linear(reduction, channel, bias=False),
        #    nn.Sigmoid()
        # )

    def forward(self, x):

        # Extract phase information
        fft_x = torch.fft.fft2(x)
        phase = torch.angle(fft_x)

        # Perform average pooling on phase
        pooled_phase = self.avg_pool(phase).view(x.size(0), self.channel)
        # pooled_amplitude = self.avg_pool(amplitude).view(x.size(0), self.channel)

        # Compute attention weights
        phase_weights = self.phase_attn(pooled_phase).view(x.size(0), self.channel, 1, 1)
        # amplitude_weights = self.amplitude_attn(pooled_amplitude).view(x.size(0), self.channel, 1, 1)

        # enhanced_phase = phase * attention_weights.expand_as(phase)
        enhanced_x = x * phase_weights.expand_as(x)

        # Combine enhanced phase with original amplitude
        # enhanced_fft_x = torch.polar(amplitude, enhanced_phase)
        # enhanced_x = torch.fft.ifft2(enhanced_fft_x).real

        return enhanced_x

