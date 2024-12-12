import torch
import torch.nn as nn


class MaskModule(nn.Module):
    """Lightweight Amplitude-Phase Masker (APM)"""

    def __init__(self, shape):
        # APM-S shape: [1, 1, h, w]
        # APM-M shape: [1, c, h, w]
        super().__init__()
        self.mask_amplitude = nn.Parameter(torch.ones(shape))
        self.mask_phase = nn.Parameter(torch.ones(shape))
        # self.amplitude_bias = nn.Parameter(torch.zeros((1,1,1,1), dtype=torch.float32))
        # self.phase_bias = nn.Parameter(torch.zeros((1,1,1,1), dtype=torch.float32))

    def initialize_masks(self, x):
        self.mask_amplitude = nn.Parameter(torch.zeros(x.shape, dtype=torch.float32).to(x.device))
        self.mask_phase = nn.Parameter(torch.zeros(x.shape, dtype=torch.float32).to(x.device))
        self.register_parameter('mask_amplitude', self.mask_amplitude)
        self.register_parameter('mask_phase', self.mask_phase)

    def reset_masks(self):
        with torch.no_grad():
            self.mask_amplitude.fill_(1.0)
            self.mask_phase.fill_(1.0)
            # self.amplitude_bias.fill_(0.0)
            # self.phase_bias.fill_(0.0)

    def forward(self, x):
        # Extract amplitude and phase information
        freq_domain = torch.fft.fftshift(torch.fft.fft2(x))
        amplitude = torch.abs(freq_domain)
        phase = torch.angle(freq_domain)

        # Constrain the value of masker to [0,1]
        mask_amplitude = torch.sigmoid(self.mask_amplitude)
        mask_phase = torch.sigmoid(self.mask_phase)

        # amplitude_bias = torch.sigmoid(self.amplitude_bias)
        # phase_bias = torch.sigmoid(self.phase_bias)

        adjusted_amplitude = mask_amplitude * amplitude  # + self.amplitude_bias
        adjusted_phase = mask_phase * phase  # + self.phase_bias

        # Combine filtered phase with filtered amplitude
        adjusted_freq = torch.polar(adjusted_amplitude, adjusted_phase)
        adjusted_x = torch.fft.ifft2(torch.fft.ifftshift(adjusted_freq)).real

        return adjusted_x
