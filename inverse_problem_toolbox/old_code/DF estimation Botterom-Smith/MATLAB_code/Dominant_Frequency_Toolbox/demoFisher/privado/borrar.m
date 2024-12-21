f0 = 3:.1:15;
fl = 5;
fu = 15;
figure
v = 1./floor(fu./f0);
plot(f0,v,':b')
hold on
plot(f0,v.*floor(fu./f0),'r')
hold off