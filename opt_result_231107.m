r1 = [42    82
    14    38
   123   146
    37    84];

r2 = [45    84
    10    32
   124   153
    35    81];

r = r1 + r2;

figure;
bar([r(3,1)/r(3,2) r(1,1)/r(1,2); r(4,1)/r(4,2) r(2,1)/r(2,2)])
legend('Control','Opto')
ylim([0 1])

disp([r(3,1)/r(3,2) r(1,1)/r(1,2) r(4,1)/r(4,2) r(2,1)/r(2,2)])
disp([r(3,1) r(3,2); r(1,1) r(1,2); r(4,1) r(4,2); r(2,1) r(2,2)])

% カイ二乗検定
kai2_test([r(1,1), r(1,2)-r(1,1)], [r(3,1), r(3,2)-r(3,1)]);
kai2_test([r(2,1), r(2,2)-r(2,1)], [r(4,1), r(4,2)-r(4,1)]);

disp('In Opto Trial')
kai2_test([r(1,1), r(1,2)-r(1,1)], [r(2,1), r(2,2)-r(2,1)]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function p = kai2_test(sample1,sample2)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%sample: [data1, data2]

sum1 = sum(sample1);
sum2 = sum(sample2);

sum_data1 = sample1(1) + sample2(1);
sum_data2 = sample1(2) + sample2(2);

sum_all = sum1 + sum2

expect1 = sum1 ./ sum_all;
expect2 = sum2 ./ sum_all;

expect11 = expect1 * sum_data1;
expect12 = expect1 * sum_data2;
expect21 = expect2 * sum_data1;
expect22 = expect2 * sum_data2;

dif11 = ((sample1(1) - expect11)^2) / expect11;
dif12 = ((sample1(2) - expect12)^2) / expect12;
dif21 = ((sample2(1) - expect21)^2) / expect21;
dif22 = ((sample2(2) - expect22)^2) / expect22;

dif_all = dif11 + dif12 + dif21 + dif22;

p = chi2cdf(dif_all,1,'upper')

end