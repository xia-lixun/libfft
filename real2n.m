% This script shows how to calculate 2n real fft using n-complex fft
% lixun
% 2020-08-04
close all; clear all; clc;

x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];   % some real-valued signal
n = length(x);                  % must be even!
n2 = n/2;                       % assume n is even
z = x(1:2:n)+j*x(2:2:n);        % complex signal of length n/2
Z = fft(z);
Ze = .5*( Z + conj([Z(1),Z(n2:-1:2)]) );        % even part
Zo = -.5*j*( Z - conj([Z(1),Z(n2:-1:2)]) );     % odd part
X = [Ze,Ze(1)] + exp(-j*2*pi/n*(0:n2)).*[Zo,Zo(1)];  % combine
X2 = fft(x);
max(abs(X-X2(1:n2+1)))  % 2.4492e-15