function filtered_signal = matched_filter(signal, conf, mf_length)
% Create a root-raised cosine filter and filter the signal with it.
% os_factor is the oversampling factor (4 in our course)
% mf_length is the one-sided filter length. The total number of filter coefficients is 2*mf_length + 1, that is the center tap and mf_length taps to both sides.


h = rrc(conf.os_factor, conf.rolloff_factor, mf_length);
filtered_signal = conv(h, signal, 'full');