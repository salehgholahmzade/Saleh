function [rxbits, conf] = rx(rxsignal,conf)
% OFDM Receiver
%
%   [txsignal conf] = tx(txbits,conf) implements a complete causal
%   OFDM receiver.
%
%   rxsignal    : received signal
%   conf        : configuration structure
%
%   Outputs
%
%   rxbits      : received bits
%   conf        : configuration structure
%

rx_filterlen = conf.tx_filterlen;

% Downconversion
time = 0:(1/conf.f_s):length(rxsignal)/conf.f_s-(1/conf.f_s); 
rxsignal_downcv = rxsignal .* exp(-1j*2*pi*conf.f_c*time.'); 

% Lowpass filter
f_lowpass =1.05* ceil((conf.N + 1)/2)*conf.f_sp;
filtered_rxsignal = ofdmlowpass(rxsignal_downcv,conf,f_lowpass); 

% Detect preamble
% Do not keep matched filter result as it will destroy OFDM symbols
data_start = frame_sync(matched_filter(filtered_rxsignal,conf,rx_filterlen),conf);

% Sampling & faze synchronization
payload_data = zeros(conf.nsyms*conf.N,1); % in symbols
data_idx = zeros(conf.nsyms,1);

data_idx(1) = data_start;
theta_hat = zeros(conf.N, conf.nsyms);
mag_hat = zeros(conf.N,conf.nsyms);

for k = 0 : conf.nsyms-1
    
    idx_sym_start = (data_start + conf.os_factor*conf.cp) + k*conf.os_factor*(conf.N+conf.cp);
    idx_sym_end = idx_sym_start + conf.os_factor*conf.N-1;
    
    idx_payload_start = 1 + k*conf.N;
    idx_payload_end = idx_payload_start + conf.N - 1;
    
    payload_data(idx_payload_start:idx_payload_end) = osfft(filtered_rxsignal(idx_sym_start:idx_sym_end),conf.os_factor);
    
    if (k == 0)
        theta_hat(:,1) = mod(angle(payload_data(1:conf.N)) - angle(conf.training),2*pi);
        mag_hat(:,1) = abs(payload_data(1:conf.N)) - abs(conf.training);
    else
        % Continuous phase tracking
        
        % Phase estimation
        % Apply viterbi-viterbi algorithm
        deltaTheta = 1/4*angle(-payload_data(idx_payload_start:idx_payload_end).^4) + pi/2*(-1:4);
    
        % Unroll phase
        [~,ind] = min(abs(deltaTheta - theta_hat(:,k)),[],2);
        theta = deltaTheta(ind); 
        
        % Lowpass filter phase
        theta_hat(:,k+1) = mod(0.01*theta + 0.99*theta_hat(:,k), 2*pi); 
        
        figure(8)
        plot(payload_data(idx_payload_start:idx_payload_end),'.');
        title('Constellation before phase correction');
        hold on
        
        % Phase correction
        payload_data(idx_payload_start:idx_payload_end) = (payload_data(idx_payload_start:idx_payload_end).*exp(-1j * theta_hat(:,k+1)));

        figure(9)
        plot(payload_data(idx_payload_start:idx_payload_end),'.');
        title('Constellation after phase correction');
        hold on
        
        % Magnitude
        mag_hat(:,k+1) = abs(payload_data(idx_payload_start:idx_payload_end)) - abs(conf.symbols(idx_payload_start:idx_payload_end));
    end
    
    
    
    figure(10)
    plot((1:conf.N)*conf.f_sp,mod(theta_hat(:,k+1),2*pi))
    title('Channel angle vs frequency');
    hold on
    figure(11)
    plot((1:conf.N)*conf.f_sp,mag_hat(:,k+1))
    title('Channel magnitude vs frequency');
    hold on   
    figure(7)
    plot((1:conf.N)*(1/conf.f_s),abs(ifft(mag_hat(:,k+1).*exp(1j*theta_hat(:,k+1)))));
    title('Channel impulse response vs time');
    hold on 
    
    
end

% Demapping
rxbits = demapper(payload_data(conf.N+1:end),conf);

    
    



