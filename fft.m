f = figure;
N=6;
x=[6 8 5 4 5 6];

t=0:N-1;
subplot(311)
stem(t,x);
xlabel('Time (s)');
ylabel('Amplitude');
title('Input sequence')

subplot(312); 
stem(0:N-1,abs(fft(x)));  
xlabel('Frequency');
ylabel('|X(k)|');
title('Magnitude Response'); 

subplot(313); 
stem(0:N-1,angle(fft(x)));
xlabel('Frequency');
ylabel('Phase');
title('Phase Response'); 

saveas(f, 'fft_m.png');