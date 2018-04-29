function [txsignal, conf] = tx(txbits,conf)
% Digital Transmitter
%
%   [txsignal conf] = tx(txbits,conf,k) implements a complete transmitter
%
%   txbits  : Information bits
%   conf    : Universal configuration structure


%% Preamble
conf.rolloff_factor = 0.22;
conf.tx_filterlen = 10*conf.os_factor;

% Upsample preamble
preamble_upsampled = zeros(1,conf.npreamble*conf.os_factor_preamble);
preamble_upsampled(1:conf.os_factor_preamble:end) = conf.preamble.';

% Create RRC pulse 
pulse = rrc(conf.os_factor, conf.rolloff_factor, conf.tx_filterlen);

% Shape preamble
preamble_shaped = conv(preamble_upsampled,pulse.','full'); 
preamble_shaped = preamble_shaped((length(pulse)-1)/2:end-(length(pulse)-1)/2); % cut tails


%% Data

% Map QPSK
qpsk_temp = 2*(txbits-0.5); % Map 1 to -1 and 0 to 1
qpsk_symbols = 1/sqrt(2)*(qpsk_temp(1:2:end) + 1i*qpsk_temp(2:2:end)); 


% Add the BPSK training OFDM symbol
symbols = [conf.training;qpsk_symbols];
conf.symbols = symbols; % Save the symbols in the general conf structure. This will allow us to analyze the channel.

ofdm_sym_length = (conf.cp + conf.N)*conf.os_factor; % length of symbol with cyclic prefix
ofdm = zeros(1,ofdm_sym_length*conf.nsyms);

for i = 0:(conf.nsyms-1)
    
    %ofdm indices
    idx_cp_start = 1 + i*ofdm_sym_length;
    idx_cp_end = i*ofdm_sym_length + conf.cp * conf.os_factor;
    idx_sym_start  = idx_cp_end + 1;
    idx_sym_end = idx_cp_end + conf.N*conf.os_factor;
    
    %symbol indices
    idx_data_start = 1 + i*conf.N;
    idx_data_end = (i+1)*conf.N;
    
    ofdm(idx_sym_start:idx_sym_end) = osifft(symbols(idx_data_start:idx_data_end),conf.os_factor);
    ofdm(idx_cp_start:idx_cp_end) = ofdm(idx_sym_end-(idx_cp_end-idx_cp_start):idx_sym_end); %cp
end

% Normalize energy
peakvalue_preamble = max(abs(preamble_shaped));
peakvalue_ofdm = max(abs(ofdm));
txsignalofdm = [preamble_shaped./(peakvalue_preamble), ofdm./(peakvalue_ofdm)];

% Upconversion
time = 0:(1/conf.f_s):length(txsignalofdm)/conf.f_s-(1/conf.f_s);
txsignal = real(txsignalofdm.* exp(1j*2*pi*conf.f_c*time));
