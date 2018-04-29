% % % % %
% Wireless Receivers: algorithms and architectures
% OFDM Audio Transmission Framework 
%
%
%   3 operating modes:
%   - 'matlab' : generic MATLAB audio routines (unreliable under Linux)
%   - 'native' : OS native audio system
%       - ALSA audio tools, most Linux distrubtions
%       - builtin WAV tools on Windows 
%   - 'bypass' : no audio transmission, takes txsignal as received signal
%
% Saleh Gholam Zadeh & Michiel Van Beirendonck

close all
clear all
clc




%% Configuration Values
conf.audiosystem = 'matlab'; % Values: 'matlab','native','bypass'
conf.payload = 'randbits'; % Values: 'image','randbits'

conf.f_s = 48000;           % sampling rate  
conf.N = 256;               % nb of carriers
conf.f_sp = 5;              % spacing frequency
conf.cp = conf.N/2;         % length of the cyclic prefix
conf.f_c = 8000;            % Central carrier frequency
conf.f_sym = 500;           % symbol frequency for the preamble

conf.nbits   = 10*conf.N;  % number of bits for randbits mode
conf.nframes = 1;           % number of frames for randbits mode

conf.npreamble  = 100;  % length of the preamble
conf.bitsps     = 16;   % bits per audio sample

%% Init Section
% All calculations we have to do only once



if strcmp(conf.payload,'image')
    I = imread('coins.png'); % Read image
    BW = imbinarize(I,0.5); % Image to binary (black/white)
    BW = BW(:,1:256); % Reshape to fit in integer number of OFDM symbols
    txbits = reshape(BW,[],1); % Reshape into bit vector 
    conf.nbits = length(txbits);
    conf.nframes = 1; % TODO: could send image in multiple frames
end


conf.nsyms = ceil(conf.nbits/(conf.N*2)) + 1; % number of OFDM symbols (+1 for training symbol)

% Initialize result structure with zero
res.biterrors   = zeros(conf.nframes,1);
res.rxnbits     = zeros(conf.nframes,1);

% Generate preamble, map it to BPSK
conf.preamble = randi([0 1],conf.npreamble,1);
conf.preamble = 1-2*conf.preamble;

% Generate training bits, map them to BPSK
conf.training = randi([0 1],conf.N,1);
conf.training = 1-2*conf.training;


conf.os_factor = conf.f_s/(conf.f_sp*conf.N); % os_factor for OFDM
conf.os_factor_preamble = conf.f_s/conf.f_sym; % os_factor for the preamble
if mod(conf.os_factor_preamble,1) ~= 0
       disp('WARNING: Sampling rate must be a multiple of the symbol rate'); 
end




%% Audio Transmission Framework
   
for k=1:conf.nframes % simulate a number of frames in randbits mode

    if strcmp(conf.payload,'randbits')
        txbits = randi([0 1],conf.nbits,1);
    end

    % tx() Transmit Function
    [txsignal, conf] = tx(txbits,conf);
    
    % Plot transmitted signal for debugging
    figure;
    plot(txsignal);
    title('Transmitted Signal')

    % % % % % % % % % % % %
    % Begin
    % Audio Transmission
    %

    % normalize values
    %peakvalue       = max(abs(txsignal));
    %normtxsignal    = txsignal / (peakvalue + 0.3);

    % create vector for transmission
    rawtxsignal = [ zeros(conf.f_s,1) ; txsignal.' ;  zeros(conf.f_s,1) ]; % padding
    rawtxsignal = [  rawtxsignal  zeros(size(rawtxsignal)) ]; % add second channel: no signal
    txdur       = length(rawtxsignal)/conf.f_s; % calculate length of transmitted signal

    % wavwrite(rawtxsignal,conf.f_s,16,'out.wav')   
    audiowrite('out.wav',rawtxsignal,conf.f_s)  

    % Platform native audio mode 
    if strcmp(conf.audiosystem,'native')

        % Windows WAV mode 
        if ispc()
            disp('Windows WAV');
            wavplay(rawtxsignal,conf.f_s,'async');
            disp('Recording in Progress');
            rawrxsignal = wavrecord((txdur+1)*conf.f_s,conf.f_s);
            disp('Recording complete')
            rxsignal = rawrxsignal(1:end,1);

        % ALSA WAV mode 
        elseif isunix()
            disp('Linux ALSA');
            cmd = sprintf('arecord -c 2 -r %d -f s16_le  -d %d in.wav &',conf.f_s,ceil(txdur)+1);
            system(cmd); 
            disp('Recording in Progress');
            system('aplay  out.wav')
            pause(2);
            disp('Recording complete')
            rawrxsignal = wavread('in.wav');
            rxsignal    = rawrxsignal(1:end,1);
        end

    % MATLAB audio mode
    elseif strcmp(conf.audiosystem,'matlab')
        disp('MATLAB generic');
        playobj = audioplayer(rawtxsignal,conf.f_s,conf.bitsps);
        recobj  = audiorecorder(conf.f_s,conf.bitsps,1);
        record(recobj);
        disp('Recording in Progress');
        playblocking(playobj)
        pause(0.5);
        stop(recobj);
        disp('Recording complete')
        rawrxsignal  = getaudiodata(recobj,'int16');
        rxsignal     = double(rawrxsignal(1:end))/double(intmax('int16')) ;

    elseif strcmp(conf.audiosystem,'bypass')
        rawrxsignal = rawtxsignal(:,1);
        rxsignal    = rawrxsignal;
    end

    % Plot received signal for debugging
    figure;
    plot(rxsignal);
    title('Received Signal')

    %
    % End
    % Audio Transmission   
    % % % % % % % % % % % %

    % rx() Receive Function
    [rxbits, conf]       = rx(rxsignal,conf);



    res.rxnbits(k)      = length(rxbits);  
    res.biterrors(k)    = sum(rxbits ~= txbits);

end

per = sum(res.biterrors > 0)/conf.nframes % phrame error rate
ber = sum(res.biterrors)/sum(res.rxnbits) % bit error rate

if strcmp(conf.payload,'image')
    figure
    imshowpair(I(:,1:256),reshape(BW,246,256),'montage')
    figure
    imshowpair(reshape(BW,246,256),reshape(rxbits,246,256),'montage')
end;
