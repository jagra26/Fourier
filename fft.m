f = figure;
N=6;
x=[6 8 5 4 5 6];

t=0:N-1;
subplot(411)
stem(t,x);
xlabel('Time (s)');
ylabel('Amplitude');
title('Input sequence')

subplot(412); 
stem(0:N-1,abs(fft(x)));  
xlabel('Frequency');
ylabel('|X(k)|');
title('Magnitude Response'); 

subplot(413); 
stem(0:N-1,angle(fft(x)));
xlabel('Frequency');
ylabel('Phase');
title('Phase Response'); 

subplot(414)
stem(t,ifft(fft(x)));
xlabel('Time (s)');
ylabel('Amplitude');
title('Inverse transform sequence')

saveas(f, 'fft_m.png');